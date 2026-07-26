from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .economics import (
    DEFAULT_PRICE_MICRO_PER_WORK_UNIT,
    ECONOMICS_VERSION,
    validate_sponsored_price,
)
from .executor import NativeWorker
from .join_protocol import (
    ASSIGNMENT_DOMAIN,
    HEARTBEAT_DOMAIN,
    INFERENCE_ACCESS_DOMAIN,
    JOIN_PROTOCOL,
    LEAVE_DOMAIN,
    READY_DOMAIN,
    REGISTRATION_DOMAIN,
    STATUS_DOMAIN,
    WORK_TICKET_DOMAIN,
    now_ms,
    request_json,
    sign_payload,
    verify_payload,
)
from .util import canonical_json, sha256, utc_now
from .wallet import Wallet, address_from_public_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_BODY_BYTES = 1_000_000


class ProviderNode:
    """Provider-owned identity, RPC backend, and signed work endpoint."""

    def __init__(
        self,
        *,
        coordinator_url: str,
        node_id: str,
        name: str,
        operator: str,
        region: str,
        memory_gib: float,
        network_latency_ms: int,
        executor_path: Path,
        rpc_binary: Path | None,
        data_dir: Path,
        listen_host: str = "0.0.0.0",
        advertise_host: str = "127.0.0.1",
        api_port: int = 51800,
        rpc_port: int = 50052,
        heartbeat_seconds: float = 10,
        coordinator_public_key: str | None = None,
        price_micro_per_work_unit: int = DEFAULT_PRICE_MICRO_PER_WORK_UNIT,
        role: str = "dedicated-worker",
    ):
        if role not in {"interactive-user", "dedicated-worker"}:
            raise ValueError(
                "provider role must be interactive-user or dedicated-worker"
            )
        self.coordinator_url = coordinator_url.rstrip("/")
        self.node_id = node_id
        self.name = name
        self.operator = operator
        self.region = region
        self.memory_gib = float(memory_gib)
        self.network_latency_ms = int(network_latency_ms)
        self.executor_path = executor_path
        self.rpc_binary = rpc_binary
        self.data_dir = data_dir
        self.listen_host = listen_host
        self.advertise_host = advertise_host
        self.api_port = api_port
        self.rpc_port = rpc_port
        self.heartbeat_seconds = heartbeat_seconds
        self.role = role
        self.price_micro_per_work_unit = validate_sponsored_price(
            price_micro_per_work_unit
        )
        self.assignment: dict | None = None
        self.assignment_signature = ""
        pin_path = self.data_dir / "coordinator_public_key.txt"
        stored_coordinator_key = (
            pin_path.read_text(encoding="utf-8").strip()
            if pin_path.exists()
            else ""
        )
        if (
            coordinator_public_key
            and stored_coordinator_key
            and coordinator_public_key != stored_coordinator_key
        ):
            raise RuntimeError("configured coordinator key does not match stored pin")
        self.coordinator_public_key = (
            coordinator_public_key or stored_coordinator_key
        )
        self.balance = 0.0
        self.joined = False
        self.last_heartbeat_at: str | None = None
        self.last_error: str | None = None
        self.completed_jobs = 0
        self.chain_height = 0
        self._rpc_process: subprocess.Popen[bytes] | None = None
        self._rpc_log: object | None = None
        self._http_server: ProviderHttpServer | None = None
        self._http_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ticket_lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self._active_jobs: dict[str, dict] = {}
        self._recent_jobs: deque[dict] = deque(maxlen=100)
        self._used_challenges: set[str] = set()
        self._challenge_order: deque[str] = deque()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        operator_token_path = self.data_dir / "operator_token"
        if operator_token_path.exists():
            self.operator_token = operator_token_path.read_text(
                encoding="utf-8"
            ).strip()
        else:
            self.operator_token = secrets.token_hex(32)
            operator_token_path.write_text(
                self.operator_token + "\n", encoding="utf-8"
            )
            operator_token_path.chmod(0o600)
        self.browser_token = secrets.token_urlsafe(32)
        coordinator = urlparse(self.coordinator_url)
        self.browser_origin = f"{coordinator.scheme}://{coordinator.netloc}"
        self.wallet = Wallet.load_or_create(self.data_dir / "node_wallet.json")
        self.worker = NativeWorker(
            self.node_id, self.executor_path, wallet=self.wallet
        )

    @property
    def reward_address(self) -> str:
        return address_from_public_key(self.worker.public_key)

    @property
    def rpc_endpoint(self) -> str:
        return f"{self.advertise_host}:{self.rpc_port}"

    @property
    def attestation_endpoint(self) -> str:
        return f"http://{self.advertise_host}:{self.api_port}"

    @staticmethod
    def _available_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind((host, 0))
            return int(reservation.getsockname()[1])

    @staticmethod
    def _wait_for_port(host: str, port: int, process: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama.cpp RPC server exited with status {process.returncode}"
                )
            try:
                with socket.create_connection((host, port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("llama.cpp RPC server did not become reachable")

    def _start_rpc(self) -> None:
        if self.rpc_binary is None:
            if self.rpc_port == 0:
                self.rpc_port = 1
            return
        if not self.rpc_binary.is_file():
            raise RuntimeError(f"RPC binary does not exist: {self.rpc_binary}")
        if self.rpc_port == 0:
            self.rpc_port = self._available_port(self.listen_host)
        self._rpc_log = (self.data_dir / "rpc.log").open("ab")
        environment = os.environ.copy()
        environment["LLAMA_CACHE"] = str(self.data_dir / "tensor-cache")
        self._rpc_process = subprocess.Popen(
            [
                str(self.rpc_binary),
                "--host",
                self.listen_host,
                "--port",
                str(self.rpc_port),
                "--threads",
                "2",
                "--cache",
            ],
            stdout=self._rpc_log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        probe_host = (
            "127.0.0.1" if self.listen_host in {"0.0.0.0", "::"} else self.listen_host
        )
        self._wait_for_port(probe_host, self.rpc_port, self._rpc_process)

    def start(self) -> dict:
        self._start_rpc()
        self._http_server = ProviderHttpServer(
            (self.listen_host, self.api_port), self
        )
        self.api_port = int(self._http_server.server_port)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name=f"{self.node_id}-provider-api",
            daemon=True,
        )
        self._http_thread.start()
        result = self.join_network()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"{self.node_id}-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        return result

    def _registration_payload(self) -> dict:
        return {
            "protocol": JOIN_PROTOCOL,
            "nodeId": self.node_id,
            "name": self.name,
            "operator": self.operator,
            "region": self.region,
            "role": self.role,
            "memoryGiB": self.memory_gib,
            "networkLatencyMs": self.network_latency_ms,
            "priceMicroPerWorkUnit": self.price_micro_per_work_unit,
            "rpcEndpoint": self.rpc_endpoint,
            "attestationEndpoint": self.attestation_endpoint,
            "capabilities": list(self.worker.capabilities),
            "publicKey": self.worker.public_key,
            "timestampMs": now_ms(),
            "nonce": secrets.token_hex(16),
        }

    def join_network(self) -> dict:
        registration = self._registration_payload()
        response = request_json(
            f"{self.coordinator_url}/api/nodes/register",
            payload={
                "registration": registration,
                "signature": sign_payload(
                    self.wallet, REGISTRATION_DOMAIN, registration
                ),
            },
        )
        assignment = response["assignment"]
        coordinator_public_key = response["coordinatorPublicKey"]
        assignment_signature = response["coordinatorSignature"]
        if (
            self.coordinator_public_key
            and coordinator_public_key != self.coordinator_public_key
        ):
            raise RuntimeError("coordinator identity does not match the pinned key")
        if not verify_payload(
            coordinator_public_key,
            ASSIGNMENT_DOMAIN,
            assignment,
            assignment_signature,
        ):
            raise RuntimeError("coordinator returned an invalid assignment signature")
        if (
            assignment["nodeId"] != self.node_id
            or assignment["publicKey"] != self.worker.public_key
            or assignment["role"] != self.role
            or assignment["rewardAddress"] != self.reward_address
            or assignment["runtime"]["rpcEndpoint"] != self.rpc_endpoint
            or assignment["economics"]["version"] != ECONOMICS_VERSION
            or assignment["economics"]["servicePriceMicroPerWorkUnit"]
            != self.price_micro_per_work_unit
        ):
            raise RuntimeError("coordinator assignment does not match this provider")

        self.assignment = assignment
        self.assignment_signature = assignment_signature
        self.coordinator_public_key = coordinator_public_key
        coordinator_pin = self.data_dir / "coordinator_public_key.txt"
        coordinator_pin.write_text(coordinator_public_key + "\n", encoding="utf-8")
        coordinator_pin.chmod(0o600)
        (self.data_dir / "assignment.json").write_text(
            json.dumps(
                {
                    "assignment": assignment,
                    "coordinatorPublicKey": coordinator_public_key,
                    "coordinatorSignature": assignment_signature,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        ready = {
            "protocol": JOIN_PROTOCOL,
            "nodeId": self.node_id,
            "assignmentId": assignment["assignmentId"],
            "timestampMs": now_ms(),
        }
        result = request_json(
            f"{self.coordinator_url}/api/nodes/{self.node_id}/ready",
            payload={
                "ready": ready,
                "signature": sign_payload(self.wallet, READY_DOMAIN, ready),
            },
        )
        self.balance = float(result["balance"])
        self.joined = True
        self.last_heartbeat_at = utc_now()
        self.last_error = None
        return result

    def authorize_inference(
        self, prompt: str, model_id: str | None = None
    ) -> dict:
        """Sign one CortexGrid Chat request while this node is online."""
        if self.role != "interactive-user":
            raise RuntimeError(
                "dedicated workers provide compute but cannot request inference"
            )
        if not self.joined or self.assignment is None:
            raise RuntimeError(
                "CortexGrid Chat must be online as a compute participant"
            )
        request = {
            "protocol": JOIN_PROTOCOL,
            "nodeId": self.node_id,
            "assignmentId": self.assignment["assignmentId"],
            "modelId": model_id or self.assignment["model"]["id"],
            "promptDigest": sha256(prompt.strip()),
            "timestampMs": now_ms(),
            "nonce": secrets.token_hex(16),
        }
        return {
            "request": request,
            "signature": sign_payload(
                self.wallet, INFERENCE_ACCESS_DOMAIN, request
            ),
        }

    def browser_inference(
        self, prompt: str, model_id: str | None = None
    ) -> dict:
        if not isinstance(prompt, str) or len(prompt.strip()) < 2:
            raise ValueError("prompt must contain at least 2 characters")
        if len(prompt) > 8_000:
            raise ValueError("prompt exceeds the 8,000 character limit")
        payload = {
            "prompt": prompt,
            "participation": self.authorize_inference(prompt, model_id),
        }
        if model_id:
            payload["modelId"] = model_id
        return request_json(
            f"{self.coordinator_url}/api/inference",
            payload=payload,
            timeout=180,
        )

    def heartbeat(self) -> dict:
        if self.assignment is None:
            raise RuntimeError("provider has no model assignment")
        heartbeat = {
            "protocol": JOIN_PROTOCOL,
            "nodeId": self.node_id,
            "assignmentId": self.assignment["assignmentId"],
            "timestampMs": now_ms(),
        }
        result = request_json(
            f"{self.coordinator_url}/api/nodes/{self.node_id}/heartbeat",
            payload={
                "heartbeat": heartbeat,
                "signature": sign_payload(
                    self.wallet, HEARTBEAT_DOMAIN, heartbeat
                ),
            },
        )
        self.balance = float(result["balance"])
        self.completed_jobs = int(result["completedJobs"])
        self.chain_height = int(result["chainHeight"])
        self.last_heartbeat_at = utc_now()
        self.last_error = None
        return result

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            if not self.joined:
                continue
            try:
                self.heartbeat()
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                self.last_error = str(error)
                print(f"[heartbeat] {error}", file=sys.stderr)
                try:
                    self.join_network()
                    print("[heartbeat] provider re-registered", file=sys.stderr)
                except (KeyError, TypeError, ValueError, RuntimeError) as join_error:
                    self.last_error = str(join_error)
                    print(
                        f"[heartbeat] re-registration failed: {join_error}",
                        file=sys.stderr,
                    )

    def leave_network(self) -> dict:
        if self.assignment is None:
            raise RuntimeError("provider has no model assignment")
        leave = {
            "protocol": JOIN_PROTOCOL,
            "nodeId": self.node_id,
            "assignmentId": self.assignment["assignmentId"],
            "timestampMs": now_ms(),
        }
        result = request_json(
            f"{self.coordinator_url}/api/nodes/{self.node_id}/leave",
            payload={
                "leave": leave,
                "signature": sign_payload(self.wallet, LEAVE_DOMAIN, leave),
            },
        )
        self.joined = False
        return result

    def signed_status(self, challenge: str) -> dict:
        payload = {
            "protocol": JOIN_PROTOCOL,
            "nodeId": self.node_id,
            "publicKey": self.worker.public_key,
            "rpcEndpoint": self.rpc_endpoint,
            "challenge": challenge,
            "assignmentId": (
                self.assignment["assignmentId"]
                if self.assignment is not None
                else None
            ),
        }
        return {
            "status": payload,
            "signature": sign_payload(self.wallet, STATUS_DOMAIN, payload),
        }

    def _validate_ticket(self, envelope: dict, operation: str) -> dict:
        if self.assignment is None:
            raise ValueError("provider does not have a model assignment")
        ticket = envelope["ticket"]
        coordinator_public_key = envelope["coordinatorPublicKey"]
        if coordinator_public_key != self.coordinator_public_key:
            raise ValueError("work ticket is not from the assigned coordinator")
        if not verify_payload(
            coordinator_public_key,
            WORK_TICKET_DOMAIN,
            ticket,
            envelope["coordinatorSignature"],
        ):
            raise ValueError("work ticket signature is invalid")

        shard = self.assignment["shard"]
        model = self.assignment["model"]
        expected = {
            "protocol": JOIN_PROTOCOL,
            "operation": operation,
            "assignmentId": self.assignment["assignmentId"],
            "nodeId": self.node_id,
            "modelId": model["id"],
            "modelVersion": model["version"],
            "shardIndex": shard["index"],
            "weightsDigest": shard["weightsDigest"],
            "computeUnits": shard["computeUnits"],
        }
        if any(ticket.get(key) != value for key, value in expected.items()):
            raise ValueError("work ticket does not match the signed assignment")
        challenge = ticket.get("challenge")
        if not isinstance(challenge, str) or len(challenge) < 16:
            raise ValueError("work ticket challenge is invalid")
        with self._ticket_lock:
            if challenge in self._used_challenges:
                raise ValueError("work ticket challenge has already been used")
            self._used_challenges.add(challenge)
            self._challenge_order.append(challenge)
            if len(self._challenge_order) > 4096:
                self._used_challenges.discard(self._challenge_order.popleft())
        return ticket

    def execute(self, envelope: dict) -> dict:
        ticket = self._validate_ticket(envelope, "execute")
        self._record_job_start(ticket)
        try:
            receipt = self.worker.execute(
                request_id=ticket["requestId"],
                model_id=ticket["modelId"],
                model_version=ticket["modelVersion"],
                shard_index=int(ticket["shardIndex"]),
                input_digest=ticket["inputDigest"],
                weights_digest=ticket["weightsDigest"],
                compute_units=int(ticket["computeUnits"]),
                challenge=ticket["challenge"],
            )
        except Exception:
            self._record_job_end(ticket["requestId"], "failed")
            raise
        self._record_job_end(ticket["requestId"], "attested")
        return {"receipt": receipt.to_dict()}

    def attest(self, envelope: dict) -> dict:
        ticket = self._validate_ticket(envelope, "attest")
        if self._rpc_process is not None and self._rpc_process.poll() is not None:
            raise RuntimeError("provider model RPC process is not running")
        receipt = self.worker.attest(
            request_id=ticket["requestId"],
            model_id=ticket["modelId"],
            model_version=ticket["modelVersion"],
            shard_index=int(ticket["shardIndex"]),
            input_digest=ticket["inputDigest"],
            output_digest=ticket["outputDigest"],
            weights_digest=ticket["weightsDigest"],
            compute_units=int(ticket["computeUnits"]),
            challenge=ticket["challenge"],
            backend=ticket["backend"],
            kernel_micros=int(ticket["kernelMicros"]),
        )
        self._record_job_end(ticket["requestId"], "attested")
        return {"receipt": receipt.to_dict()}

    def _job_from_ticket(self, ticket: dict) -> dict:
        shard = self.assignment["shard"] if self.assignment else {}
        return {
            "requestId": ticket["requestId"],
            "modelId": ticket["modelId"],
            "modelVersion": ticket["modelVersion"],
            "stage": ticket["shardIndex"],
            "layerRange": shard.get("layerRange"),
            "computeUnits": ticket["computeUnits"],
            "inputDigest": ticket["inputDigest"],
            "status": "running",
            "startedAt": utc_now(),
        }

    def _record_job_start(self, ticket: dict) -> dict:
        job = self._job_from_ticket(ticket)
        with self._jobs_lock:
            self._active_jobs[ticket["requestId"]] = job
        return job

    def _record_job_end(self, request_id: str, status: str) -> dict:
        with self._jobs_lock:
            job = self._active_jobs.pop(request_id, None)
            if job is None:
                job = next(
                    (
                        item
                        for item in self._recent_jobs
                        if item["requestId"] == request_id
                    ),
                    {"requestId": request_id},
                )
            job["status"] = status
            job["finishedAt"] = utc_now()
            if job not in self._recent_jobs:
                self._recent_jobs.appendleft(job)
            return job

    def start_job(self, envelope: dict) -> dict:
        ticket = self._validate_ticket(envelope, "start")
        return {"status": "running", "job": self._record_job_start(ticket)}

    def cancel_job(self, envelope: dict) -> dict:
        ticket = self._validate_ticket(envelope, "cancel")
        return {
            "status": "cancelled",
            "job": self._record_job_end(ticket["requestId"], "cancelled"),
        }

    def settle_job(self, envelope: dict) -> dict:
        ticket = self._validate_ticket(envelope, "settle")
        with self._jobs_lock:
            job = next(
                (
                    item
                    for item in self._recent_jobs
                    if item["requestId"] == ticket["requestId"]
                ),
                None,
            )
            if job is None:
                job = self._job_from_ticket(ticket)
                self._recent_jobs.appendleft(job)
            job.update(
                {
                    "status": "finalized",
                    "blockHeight": ticket["blockHeight"],
                    "blockHash": ticket["blockHash"],
                    "proofRoot": ticket["proofRoot"],
                    "rewardMicro": ticket["rewardMicro"],
                    "reward": ticket["reward"],
                    "settledAt": utc_now(),
                }
            )
            self.balance = float(ticket["providerBalance"])
            self.completed_jobs = int(ticket["completedJobs"])
            self.chain_height = int(ticket["blockHeight"])
            return {"status": "finalized", "job": dict(job)}

    def jobs_snapshot(self) -> dict:
        with self._jobs_lock:
            return {
                "active": [dict(job) for job in self._active_jobs.values()],
                "recent": [dict(job) for job in self._recent_jobs],
            }

    def snapshot(self) -> dict:
        jobs = self.jobs_snapshot()
        return {
            "nodeId": self.node_id,
            "name": self.name,
            "operator": self.operator,
            "region": self.region,
            "role": self.role,
            "joined": self.joined,
            "publicKey": self.worker.public_key,
            "rewardAddress": self.reward_address,
            "balance": self.balance,
            "priceMicroPerWorkUnit": self.price_micro_per_work_unit,
            "completedJobs": self.completed_jobs,
            "chainHeight": self.chain_height,
            "lastHeartbeatAt": self.last_heartbeat_at,
            "lastError": self.last_error,
            "rpcEndpoint": self.rpc_endpoint,
            "rpcOnline": (
                self._rpc_process is None or self._rpc_process.poll() is None
            ),
            "attestationEndpoint": self.attestation_endpoint,
            "assignment": self.assignment,
            "activeJobs": jobs["active"],
            "recentJobs": jobs["recent"],
        }

    def close(self) -> None:
        self._stop.set()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
            self._http_thread = None
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
        if self._rpc_process is not None:
            if self._rpc_process.poll() is None:
                self._rpc_process.terminate()
                try:
                    self._rpc_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._rpc_process.kill()
                    self._rpc_process.wait(timeout=2)
            self._rpc_process = None
        if self._rpc_log is not None:
            self._rpc_log.close()
            self._rpc_log = None
        self.worker.close()


class ProviderApiHandler(BaseHTTPRequestHandler):
    server: "ProviderHttpServer"

    def log_message(self, format: str, *args: object) -> None:
        print(
            f"[provider] {self.address_string()} {format % args}",
            file=sys.stderr,
        )

    def _json(self, status: int, body: object) -> None:
        encoded = canonical_json(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin", "")
        if origin == self.server.provider.browser_origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _require_operator(self) -> None:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.provider.operator_token}"
        if not secrets.compare_digest(supplied, expected):
            raise PermissionError("operator token is required")

    def _require_browser(self) -> None:
        if self.server.provider.role != "interactive-user":
            raise PermissionError("browser chat requires an interactive node")
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.provider.browser_token}"
        if not secrets.compare_digest(supplied, expected):
            raise PermissionError("browser bridge token is required")

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin", "")
        path = urlparse(self.path).path
        if (
            origin != self.server.provider.browser_origin
            or path not in {"/v1/browser/status", "/v1/browser/inference"}
        ):
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Authorization, Content-Type"
        )
        if (
            self.headers.get("Access-Control-Request-Private-Network")
            == "true"
        ):
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Vary", "Origin")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/v1/browser/status":
                self._require_browser()
                self._json(
                    200,
                    {
                        "connected": self.server.provider.joined,
                        "provider": self.server.provider.snapshot(),
                    },
                )
            elif parsed.path == "/v1/status":
                challenge = parse_qs(parsed.query).get("challenge", [""])[0]
                if len(challenge) < 16:
                    raise ValueError("a status challenge is required")
                self._json(200, self.server.provider.signed_status(challenge))
            elif parsed.path == "/v1/balance":
                if self.server.provider.assignment is not None:
                    self.server.provider.heartbeat()
                self._json(200, self.server.provider.snapshot())
            elif parsed.path == "/v1/provider":
                self._require_operator()
                self._json(200, self.server.provider.snapshot())
            elif parsed.path == "/v1/jobs":
                self._require_operator()
                self._json(200, self.server.provider.jobs_snapshot())
            else:
                self._json(404, {"error": "not_found"})
        except PermissionError as error:
            self._json(401, {"error": "unauthorized", "message": str(error)})
        except (KeyError, TypeError, ValueError) as error:
            self._json(400, {"error": "invalid_request", "message": str(error)})
        except Exception as error:
            self._json(500, {"error": "internal_error", "message": str(error)})

    def do_POST(self) -> None:
        try:
            body = self._body()
            if self.path == "/v1/browser/inference":
                self._require_browser()
                self._json(
                    200,
                    self.server.provider.browser_inference(
                        body.get("prompt", ""),
                        body.get("modelId"),
                    ),
                )
            elif self.path == "/v1/execute":
                self._json(200, self.server.provider.execute(body))
            elif self.path == "/v1/attest":
                self._json(200, self.server.provider.attest(body))
            elif self.path == "/v1/jobs/start":
                self._json(200, self.server.provider.start_job(body))
            elif self.path == "/v1/jobs/cancel":
                self._json(200, self.server.provider.cancel_job(body))
            elif self.path == "/v1/jobs/settle":
                self._json(200, self.server.provider.settle_job(body))
            elif self.path == "/v1/operator/join":
                self._require_operator()
                self._json(200, self.server.provider.join_network())
            elif self.path == "/v1/operator/leave":
                self._require_operator()
                self._json(200, self.server.provider.leave_network())
            else:
                self._json(404, {"error": "not_found"})
        except PermissionError as error:
            self._json(401, {"error": "unauthorized", "message": str(error)})
        except (KeyError, TypeError, ValueError) as error:
            self._json(400, {"error": "invalid_request", "message": str(error)})
        except RuntimeError as error:
            self._json(503, {"error": "unavailable", "message": str(error)})
        except Exception as error:
            self._json(500, {"error": "internal_error", "message": str(error)})


class ProviderHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], provider: ProviderNode):
        self.provider = provider
        super().__init__(address, ProviderApiHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join a CortexGrid inference network")
    parser.add_argument("--coordinator", default="http://127.0.0.1:65534")
    parser.add_argument("--coordinator-public-key")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--name")
    parser.add_argument("--operator", default="Independent")
    parser.add_argument("--region", default="unknown")
    parser.add_argument("--memory-gib", type=float, required=True)
    parser.add_argument("--latency-ms", type=int, default=50)
    parser.add_argument(
        "--price-micro-per-work-unit",
        type=int,
        default=DEFAULT_PRICE_MICRO_PER_WORK_UNIT,
        help="legacy join field; ignored by CTX economics v4",
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--api-port", type=int, default=51800)
    parser.add_argument("--rpc-port", type=int, default=50052)
    parser.add_argument(
        "--executor",
        type=Path,
        default=PROJECT_ROOT / "build" / "cortex_worker",
    )
    parser.add_argument(
        "--rpc-binary",
        type=Path,
        default=PROJECT_ROOT / "build" / "llama" / "bin" / "ggml-rpc-server",
    )
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir or PROJECT_ROOT / ".provider" / args.node_id
    provider = ProviderNode(
        coordinator_url=args.coordinator,
        node_id=args.node_id,
        name=args.name or args.node_id,
        operator=args.operator,
        region=args.region,
        memory_gib=args.memory_gib,
        network_latency_ms=args.latency_ms,
        executor_path=args.executor,
        rpc_binary=args.rpc_binary,
        data_dir=data_dir,
        listen_host=args.listen_host,
        advertise_host=args.advertise_host,
        api_port=args.api_port,
        rpc_port=args.rpc_port,
        coordinator_public_key=args.coordinator_public_key,
        price_micro_per_work_unit=args.price_micro_per_work_unit,
    )
    stopped = threading.Event()

    def stop(_signal: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        result = provider.start()
        assignment = result["assignment"]["shard"]
        print(
            f"{args.node_id} joined as stage {assignment['index']} "
            f"(layers {assignment.get('layerRange')})"
        )
        print(f"Reward address: {provider.reward_address}")
        print(f"CTX balance: {result['balance']}")
        print(
            f"Operate locally: ./CortexNode status --node-id {args.node_id} "
            f"--api http://127.0.0.1:{provider.api_port}"
        )
        while not stopped.wait(1):
            pass
    finally:
        provider.close()


if __name__ == "__main__":
    main()
