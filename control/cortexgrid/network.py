from __future__ import annotations

import math
import os
import re
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlparse

from .blockchain import (
    Blockchain,
    TOKEN_SYMBOL,
    receipt_digest,
    tokens,
    verify_proof_bundle,
)
from .economics import (
    CAPACITY_TARGET_REPLICAS,
    DEFAULT_PRICE_MICRO_PER_WORK_UNIT,
    build_economics_record,
    validate_price,
)
from .executor import ExecutionReceipt, NativeWorker, verify_receipt_signature
from .join_protocol import (
    ASSIGNMENT_DOMAIN,
    HEARTBEAT_DOMAIN,
    JOIN_PROTOCOL,
    LEAVE_DOMAIN,
    READY_DOMAIN,
    REGISTRATION_DOMAIN,
    STATUS_DOMAIN,
    now_ms,
    request_json,
    require_fresh,
    sign_payload,
    verify_payload,
)
from .model_runtime import DistributedModelRuntime
from .remote_worker import RemoteWorker
from .util import merkle_root, sha256, sha256_file, utc_now
from .wallet import Wallet, address_from_public_key


NODE_BLUEPRINTS = [
    ("atlas", "Atlas", "Northstar Labs", "us-east", 24, 18, "#8da2fb"),
    ("ember", "Ember", "Independent", "us-west", 24, 31, "#ff8b5b"),
    ("kite", "Kite", "Tensor House", "eu-west", 24, 44, "#5ee3b1"),
    ("orbit", "Orbit", "Lattice DAO", "eu-central", 24, 51, "#d99cff"),
    ("sable", "Sable", "Independent", "ap-south", 24, 72, "#ffd166"),
    ("nova", "Nova", "Compute Union", "ap-northeast", 24, 81, "#60c7ff"),
]
REMOTE_HEARTBEAT_TIMEOUT_MS = 45_000
NODE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,47}")


def _synthetic_model_manifest() -> dict:
    model = {
        "id": "cortex-72b-devnet",
        "name": "Cortex 72B · Devnet",
        "version": "0.1.0",
        "parameters": 72_000_000_000,
        "precision": "4-bit",
        "architecture": "pipeline-parallel",
        "shardCount": 12,
        "shardSizeGiB": 3.5,
        "replicationFactor": 2,
        "contextWindow": 32_768,
        "status": "ready",
    }
    model["shards"] = [
        {
            "index": index,
            "sizeGiB": model["shardSizeGiB"],
            "computeUnits": 2 + index % 3,
            "weightsDigest": sha256(
                f"cortex-grid-weights|{model['id']}|{model['version']}|{index}"
            ),
        }
        for index in range(model["shardCount"])
    ]
    return model


def _real_model_manifest(model_path: Path) -> dict:
    model_digest = sha256_file(model_path)
    model_size_gib = model_path.stat().st_size / (1024**3)
    model = {
        "id": "smollm2-135m-rpc",
        "name": "SmolLM2 135M · Distributed",
        "version": f"q4km-{model_digest[:12]}",
        "parameters": 135_000_000,
        "precision": "Q4_K_M",
        "architecture": "llama.cpp-rpc-layer",
        "shardCount": 2,
        "shardSizeGiB": round(model_size_gib / 2, 6),
        "replicationFactor": 3,
        "contextWindow": 1_024,
        "status": "ready",
        "modelDigest": model_digest,
        "executionMode": "real-transformer",
    }
    model["shards"] = [
        {
            "index": index,
            "sizeGiB": model["shardSizeGiB"],
            "computeUnits": 15,
            "layerRange": [index * 15, index * 15 + 14],
            "weightsDigest": sha256(
                f"{model_digest}|pipeline-stage|{index}|layers-{index * 15}-"
                f"{index * 15 + 14}"
            ),
        }
        for index in range(model["shardCount"])
    ]
    return model


@dataclass
class ComputeNode:
    id: str
    name: str
    operator: str
    region: str
    memory_gib: float
    network_latency_ms: int
    accent: str
    worker: NativeWorker | RemoteWorker
    price_micro_per_work_unit: int = DEFAULT_PRICE_MICRO_PER_WORK_UNIT
    status: str = "online"
    completed_jobs: int = 0
    failed_proofs: int = 0
    shards: list[dict] = field(default_factory=list)
    remote: bool = False
    rpc_endpoint: str | None = None
    attestation_endpoint: str | None = None
    assignment: dict | None = None
    last_heartbeat_ms: int | None = None

    @property
    def allocated_memory_gib(self) -> float:
        return sum(shard["sizeGiB"] for shard in self.shards)

    def hosts(self, model_id: str, shard_index: int) -> bool:
        return any(
            shard["modelId"] == model_id and shard["index"] == shard_index
            for shard in self.shards
        )

    def snapshot(self, chain: Blockchain) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "operator": self.operator,
            "region": self.region,
            "status": self.status,
            "memoryGiB": self.memory_gib,
            "allocatedMemoryGiB": self.allocated_memory_gib,
            "freeMemoryGiB": self.memory_gib - self.allocated_memory_gib,
            "networkLatencyMs": self.network_latency_ms,
            "completedJobs": self.completed_jobs,
            "failedProofs": self.failed_proofs,
            "backend": self.worker.capabilities,
            "publicKey": self.worker.public_key,
            "rewardAddress": self.worker.reward_address,
            "balance": tokens(chain.balance_of(self.worker.reward_address)),
            "priceMicroPerWorkUnit": self.price_micro_per_work_unit,
            "accent": self.accent,
            "remote": self.remote,
            "rpcEndpoint": self.rpc_endpoint,
            "attestationEndpoint": self.attestation_endpoint,
            "lastHeartbeatMs": self.last_heartbeat_ms,
            "assignment": (
                self.assignment["assignment"]
                if self.assignment is not None
                else None
            ),
            "shards": [
                {
                    "modelId": shard["modelId"],
                    "index": shard["index"],
                    "sizeGiB": shard["sizeGiB"],
                    "weightsDigest": shard["weightsDigest"],
                }
                for shard in self.shards
            ],
        }


class CortexNetwork:
    """The single Python control plane: route, verify, finalize a block."""

    def __init__(
        self,
        *,
        executor_path: Path | None = None,
        data_dir: Path | None = None,
        real_model: bool | None = None,
    ):
        project_root = Path(__file__).resolve().parents[2]
        self.executor_path = executor_path or Path(
            os.environ.get(
                "CORTEX_EXECUTOR_PATH", project_root / "build" / "cortex_worker"
            )
        )
        self.data_dir = data_dir or Path(
            os.environ.get("CORTEX_DATA_DIR", project_root / ".data")
        )
        self.model_path = Path(
            os.environ.get(
                "CORTEX_MODEL_PATH",
                project_root / "models" / "SmolLM2-135M-Instruct-Q4_K_M.gguf",
            )
        )
        self.rpc_binary = Path(
            os.environ.get(
                "CORTEX_RPC_BINARY",
                project_root / "build" / "llama" / "bin" / "ggml-rpc-server",
            )
        )
        self.model_server_binary = Path(
            os.environ.get(
                "CORTEX_MODEL_SERVER_BINARY",
                project_root / "build" / "llama" / "bin" / "llama-server",
            )
        )
        runtime_available = DistributedModelRuntime.is_available(
            self.model_path, self.rpc_binary, self.model_server_binary
        )
        if real_model is None:
            runtime_setting = os.environ.get("CORTEX_REAL_MODEL", "auto").lower()
            real_model = runtime_available if runtime_setting == "auto" else (
                runtime_setting in {"1", "true", "on", "yes"}
            )
        if real_model and not runtime_available:
            raise RuntimeError(
                "real model mode requested, but the GGUF or llama.cpp binaries "
                "are missing; run `make bootstrap-model`"
            )

        self.real_model_enabled = bool(real_model)
        self.model = (
            _real_model_manifest(self.model_path)
            if self.real_model_enabled
            else _synthetic_model_manifest()
        )
        self.demo_wallet = Wallet.load_or_create(self.data_dir / "demo_wallet.json")
        self.coordinator_wallet = Wallet.load_or_create(
            self.data_dir / "coordinator_wallet.json"
        )
        chain_filename = (
            f"chain-{self.model['id']}.jsonl"
            if self.real_model_enabled
            else "chain.jsonl"
        )
        self.chain = Blockchain(
            self.data_dir / chain_filename,
            demo_address=self.demo_wallet.address,
            model_manifest=self.model,
        )
        self.nodes: dict[str, ComputeNode] = {}
        self.model_runtime: DistributedModelRuntime | None = None
        self.events: list[dict] = []
        self.recent_inferences: list[dict] = []
        self._lock = threading.RLock()
        self._closed = False

        try:
            self._start_nodes()
            self._place_shards()
            if self.real_model_enabled:
                self.model_runtime = DistributedModelRuntime(
                    model_path=self.model_path,
                    rpc_binary=self.rpc_binary,
                    server_binary=self.model_server_binary,
                    node_ids=list(self.nodes),
                    data_dir=self.data_dir / "runtime",
                )
        except Exception:
            self.close()
            raise
        self._event(
            "network",
            "Execution network online",
            f"{len(self.nodes)} native workers · {self.model['shardCount']} shards · "
            f"{self.model['replicationFactor']}× replicas",
        )

    def _start_nodes(self) -> None:
        for blueprint in NODE_BLUEPRINTS:
            node_id, name, operator, region, memory, latency, accent = blueprint
            worker = NativeWorker(node_id, self.executor_path)
            self.nodes[node_id] = ComputeNode(
                id=node_id,
                name=name,
                operator=operator,
                region=region,
                memory_gib=memory,
                network_latency_ms=latency,
                accent=accent,
                worker=worker,
            )

    def _place_shards(self) -> None:
        nodes = list(self.nodes.values())
        for shard in self.model["shards"]:
            stride = max(1, len(nodes) // self.model["replicationFactor"])
            replica_indices = [
                (shard["index"] + replica * stride) % len(nodes)
                for replica in range(self.model["replicationFactor"])
            ]
            for replica_index in replica_indices:
                nodes[replica_index].shards.append(
                    {"modelId": self.model["id"], **shard}
                )

    def _event(self, kind: str, title: str, detail: str, **metadata: object) -> None:
        self.events.insert(
            0,
            {
                "id": f"evt_{secrets.token_hex(6)}",
                "type": kind,
                "title": title,
                "detail": detail,
                "timestamp": utc_now(),
                **metadata,
            },
        )
        del self.events[100:]

    def _route(self, shard_index: int) -> ComputeNode:
        self._expire_remote_nodes()
        candidates = [
            node
            for node in self.nodes.values()
            if node.status == "online" and node.hosts(self.model["id"], shard_index)
        ]
        if not candidates:
            raise RuntimeError(f"no online replica for shard {shard_index}")
        return min(
            candidates,
            key=lambda node: (
                node.price_micro_per_work_unit,
                node.completed_jobs,
                node.network_latency_ms,
                node.id,
            ),
        )

    def _expire_remote_nodes(self) -> None:
        current = now_ms()
        for node in self.nodes.values():
            if (
                node.remote
                and node.status == "online"
                and (
                    node.last_heartbeat_ms is None
                    or current - node.last_heartbeat_ms
                    > REMOTE_HEARTBEAT_TIMEOUT_MS
                )
            ):
                node.status = "offline"
                self._event(
                    "warning",
                    f"{node.name} heartbeat expired",
                    f"{node.region} · removed from inference routing",
                    nodeId=node.id,
                )

    def _verify_current_receipt(
        self,
        receipt: ExecutionReceipt,
        node: ComputeNode,
        shard: dict,
        expected_input: str,
    ) -> None:
        checks = {
            "node identity": (
                receipt.node_id == node.id
                and receipt.public_key == node.worker.public_key
            ),
            "model identity": (
                receipt.model_id == self.model["id"]
                and receipt.model_version == self.model["version"]
            ),
            "shard assignment": (
                receipt.shard_index == shard["index"]
                and node.hosts(self.model["id"], shard["index"])
            ),
            "weight commitment": receipt.weights_digest == shard["weightsDigest"],
            "work commitment": receipt.compute_units == shard["computeUnits"],
            "activation chain": receipt.input_digest == expected_input,
            "worker signature": verify_receipt_signature(receipt),
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            node.failed_proofs += 1
            raise RuntimeError(f"rejected shard receipt: {', '.join(failed)}")

    @staticmethod
    def _validate_attestation_endpoint(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("attestationEndpoint must be a URL")
        endpoint = value.rstrip("/")
        parsed = urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("attestationEndpoint must be an HTTP(S) origin")
        return endpoint

    @staticmethod
    def _parse_rpc_endpoint(value: object) -> tuple[str, int, str]:
        if not isinstance(value, str):
            raise ValueError("rpcEndpoint must use host:port format")
        parsed = urlparse(f"//{value}")
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("rpcEndpoint contains an invalid port") from error
        if (
            not parsed.hostname
            or port is None
            or port < 1
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("rpcEndpoint must use host:port format")
        return parsed.hostname, port, value

    @staticmethod
    def _validate_public_key(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("publicKey must be a 32-byte Ed25519 key")
        try:
            raw = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError("publicKey must be hexadecimal") from error
        if len(raw) != 32:
            raise ValueError("publicKey must be a 32-byte Ed25519 key")
        return value

    @staticmethod
    def _bounded_text(
        value: object, field_name: str, *, minimum: int = 1, maximum: int = 80
    ) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be text")
        normalized = value.strip()
        if not minimum <= len(normalized) <= maximum:
            raise ValueError(
                f"{field_name} must contain between {minimum} and {maximum} characters"
            )
        return normalized

    def _probe_remote_identity(
        self,
        *,
        node_id: str,
        public_key: str,
        rpc_endpoint: str,
        attestation_endpoint: str,
        assignment_id: str | None = None,
    ) -> int:
        challenge = secrets.token_hex(16)
        started = time.perf_counter()
        response = request_json(
            f"{attestation_endpoint}/v1/status?challenge={quote(challenge)}",
            timeout=3,
        )
        status = response["status"]
        if (
            status.get("protocol") != JOIN_PROTOCOL
            or status.get("nodeId") != node_id
            or status.get("publicKey") != public_key
            or status.get("rpcEndpoint") != rpc_endpoint
            or status.get("challenge") != challenge
        ):
            raise ValueError("node status response does not match its registration")
        if assignment_id is not None and status.get("assignmentId") != assignment_id:
            raise ValueError("node has not installed the signed assignment")
        if not verify_payload(
            public_key, STATUS_DOMAIN, status, response["signature"]
        ):
            raise ValueError("node status signature is invalid")
        return max(1, round((time.perf_counter() - started) * 1_000))

    def _choose_remote_shard(self, memory_gib: float) -> dict:
        compatible = [
            shard
            for shard in self.model["shards"]
            if float(shard["sizeGiB"]) <= memory_gib
        ]
        if not compatible:
            minimum = min(
                float(shard["sizeGiB"]) for shard in self.model["shards"]
            )
            raise ValueError(
                f"node needs at least {minimum} GiB for an available stage"
            )
        return min(
            compatible,
            key=lambda shard: (
                sum(
                    node.hosts(self.model["id"], shard["index"])
                    for node in self.nodes.values()
                ),
                shard["index"],
            ),
        )

    def register_remote_node(self, envelope: dict) -> dict:
        registration = envelope["registration"]
        signature = envelope["signature"]
        if registration.get("protocol") != JOIN_PROTOCOL:
            raise ValueError("unsupported node join protocol")
        node_id = registration.get("nodeId")
        if not isinstance(node_id, str) or not NODE_ID_PATTERN.fullmatch(node_id):
            raise ValueError(
                "nodeId must be 3-48 lowercase letters, digits, dashes, or underscores"
            )
        name = self._bounded_text(registration.get("name"), "name")
        operator = self._bounded_text(registration.get("operator"), "operator")
        region = self._bounded_text(registration.get("region"), "region")
        public_key = self._validate_public_key(registration.get("publicKey"))
        rpc_host, rpc_port, rpc_endpoint = self._parse_rpc_endpoint(
            registration.get("rpcEndpoint")
        )
        attestation_endpoint = self._validate_attestation_endpoint(
            registration.get("attestationEndpoint")
        )
        capabilities = registration.get("capabilities")
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or not all(
                isinstance(item, str) and 1 <= len(item) <= 40
                for item in capabilities
            )
        ):
            raise ValueError("capabilities must contain one or more backend names")
        if isinstance(registration.get("memoryGiB"), bool):
            raise ValueError("memoryGiB must be numeric")
        try:
            memory_gib = float(registration["memoryGiB"])
            latency_ms = int(registration["networkLatencyMs"])
            price_micro_per_work_unit = validate_price(
                registration["priceMicroPerWorkUnit"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("node capacity values are invalid") from error
        if not math.isfinite(memory_gib) or memory_gib <= 0 or memory_gib > 1_000_000:
            raise ValueError("memoryGiB is outside the accepted range")
        if latency_ms < 0 or latency_ms > 60_000:
            raise ValueError("networkLatencyMs is outside the accepted range")
        require_fresh(registration.get("timestampMs"))
        if not isinstance(registration.get("nonce"), str) or len(
            registration["nonce"]
        ) < 16:
            raise ValueError("registration nonce is invalid")
        if not verify_payload(
            public_key, REGISTRATION_DOMAIN, registration, signature
        ):
            raise ValueError("node registration signature is invalid")

        measured_latency_ms = self._probe_remote_identity(
            node_id=node_id,
            public_key=public_key,
            rpc_endpoint=rpc_endpoint,
            attestation_endpoint=attestation_endpoint,
        )

        with self._lock:
            existing = self.nodes.get(node_id)
            if existing is not None and not existing.remote:
                raise ValueError("nodeId is reserved by a built-in participant")
            if (
                existing is not None
                and existing.worker.public_key != public_key
            ):
                raise ValueError("nodeId already belongs to a different public key")
            for other in self.nodes.values():
                if other.id == node_id:
                    continue
                if other.worker.public_key == public_key:
                    raise ValueError("publicKey is already registered to another node")
                if other.remote and (
                    other.rpc_endpoint == rpc_endpoint
                    or other.attestation_endpoint == attestation_endpoint
                ):
                    raise ValueError("provider endpoint is already registered")
            shard = (
                existing.shards[0]
                if existing is not None and existing.shards
                else {
                    "modelId": self.model["id"],
                    **self._choose_remote_shard(memory_gib),
                }
            )
            if memory_gib < float(shard["sizeGiB"]):
                raise ValueError(
                    f"node needs at least {shard['sizeGiB']} GiB for this stage"
                )
            assignment_id = sha256(
                f"{JOIN_PROTOCOL}|{node_id}|{public_key}|{self.model['version']}|"
                f"{shard['index']}|{registration['nonce']}|"
                f"{registration['timestampMs']}"
            )
            assignment = {
                "protocol": JOIN_PROTOCOL,
                "assignmentId": assignment_id,
                "issuedAt": utc_now(),
                "nodeId": node_id,
                "publicKey": public_key,
                "rewardAddress": address_from_public_key(public_key),
                "model": {
                    "id": self.model["id"],
                    "name": self.model["name"],
                    "version": self.model["version"],
                    "modelDigest": self.model.get("modelDigest"),
                    "precision": self.model["precision"],
                    "contextWindow": self.model["contextWindow"],
                },
                "shard": {
                    key: shard[key]
                    for key in (
                        "index",
                        "sizeGiB",
                        "computeUnits",
                        "weightsDigest",
                        "layerRange",
                    )
                    if key in shard
                },
                "runtime": {
                    "distributionMode": (
                        "llama.cpp-rpc-layer"
                        if self.real_model_enabled
                        else "deterministic-stage"
                    ),
                    "rpcEndpoint": rpc_endpoint,
                    "tensorCache": "provider-owned",
                },
                "economics": {
                    "priceMicroPerWorkUnit": price_micro_per_work_unit,
                    "priceUnit": "micro-CTX/model-compute-unit-weighted-token",
                },
                "lease": {
                    "heartbeatSeconds": 10,
                    "timeoutSeconds": REMOTE_HEARTBEAT_TIMEOUT_MS // 1_000,
                },
            }
            assignment_signature = sign_payload(
                self.coordinator_wallet, ASSIGNMENT_DOMAIN, assignment
            )
            worker = RemoteWorker(
                node_id=node_id,
                public_key=public_key,
                capabilities=capabilities,
                endpoint=attestation_endpoint,
                assignment_id=assignment_id,
                coordinator_wallet=self.coordinator_wallet,
            )
            node = ComputeNode(
                id=node_id,
                name=name,
                operator=operator,
                region=region,
                memory_gib=memory_gib,
                network_latency_ms=measured_latency_ms,
                accent="#4dd4ac",
                worker=worker,
                price_micro_per_work_unit=price_micro_per_work_unit,
                status="assigned",
                completed_jobs=existing.completed_jobs if existing else 0,
                failed_proofs=existing.failed_proofs if existing else 0,
                shards=[{"modelId": self.model["id"], **shard}],
                remote=True,
                rpc_endpoint=rpc_endpoint,
                attestation_endpoint=attestation_endpoint,
                assignment={
                    "assignment": assignment,
                    "signature": assignment_signature,
                },
            )
            self.nodes[node_id] = node
            if self.model_runtime is not None:
                self.model_runtime.register_remote(node_id, rpc_endpoint)
            self._event(
                "node",
                f"{name} registered",
                f"{region} · assigned model stage {shard['index']}",
                nodeId=node_id,
            )
            return {
                "status": "assigned",
                "assignment": assignment,
                "coordinatorPublicKey": self.coordinator_wallet.public_key_hex,
                "coordinatorSignature": assignment_signature,
                "rewardAddress": worker.reward_address,
                "balance": tokens(self.chain.balance_of(worker.reward_address)),
            }

    def _remote_node_message(
        self, node_id: str, envelope: dict, message_key: str, domain: str
    ) -> tuple[ComputeNode, dict]:
        node = self.nodes.get(node_id)
        if node is None or not node.remote or node.assignment is None:
            raise ValueError(f"unknown remote node: {node_id}")
        payload = envelope[message_key]
        if (
            payload.get("protocol") != JOIN_PROTOCOL
            or payload.get("nodeId") != node_id
            or payload.get("assignmentId")
            != node.assignment["assignment"]["assignmentId"]
        ):
            raise ValueError("node message does not match its assignment")
        require_fresh(payload.get("timestampMs"))
        if not verify_payload(
            node.worker.public_key, domain, payload, envelope["signature"]
        ):
            raise ValueError("node message signature is invalid")
        return node, payload

    def mark_remote_ready(self, node_id: str, envelope: dict) -> dict:
        with self._lock:
            node, _ = self._remote_node_message(
                node_id, envelope, "ready", READY_DOMAIN
            )
            assignment = node.assignment["assignment"]
            self._probe_remote_identity(
                node_id=node.id,
                public_key=node.worker.public_key,
                rpc_endpoint=node.rpc_endpoint or "",
                attestation_endpoint=node.attestation_endpoint or "",
                assignment_id=assignment["assignmentId"],
            )
            if self.real_model_enabled:
                rpc_host, rpc_port, _ = self._parse_rpc_endpoint(
                    node.rpc_endpoint
                )
                try:
                    with socket.create_connection((rpc_host, rpc_port), timeout=2):
                        pass
                except OSError as error:
                    raise RuntimeError(
                        "assigned llama.cpp RPC endpoint is not reachable"
                    ) from error
            node.status = "online"
            node.last_heartbeat_ms = now_ms()
            self._event(
                "node",
                f"{node.name} is ready",
                f"stage {assignment['shard']['index']} · "
                f"{node.worker.reward_address}",
                nodeId=node.id,
            )
            return {
                "status": "ready",
                "assignment": assignment,
                "rewardAddress": node.worker.reward_address,
                "balance": tokens(
                    self.chain.balance_of(node.worker.reward_address)
                ),
            }

    def heartbeat_remote_node(self, node_id: str, envelope: dict) -> dict:
        with self._lock:
            node, _ = self._remote_node_message(
                node_id, envelope, "heartbeat", HEARTBEAT_DOMAIN
            )
            if node.status == "left":
                raise ValueError("provider must re-register before heartbeating")
            node.status = "online"
            node.last_heartbeat_ms = now_ms()
            return {
                "status": "online",
                "nodeId": node.id,
                "rewardAddress": node.worker.reward_address,
                "balance": tokens(
                    self.chain.balance_of(node.worker.reward_address)
                ),
                "completedJobs": node.completed_jobs,
                "chainHeight": len(self.chain.blocks) - 1,
            }

    def leave_remote_node(self, node_id: str, envelope: dict) -> dict:
        with self._lock:
            node, _ = self._remote_node_message(
                node_id, envelope, "leave", LEAVE_DOMAIN
            )
            node.status = "left"
            node.last_heartbeat_ms = None
            self._event(
                "node",
                f"{node.name} left the network",
                f"stage {node.shards[0]['index']} · rewards remain spendable",
                nodeId=node.id,
            )
            return {
                "status": "left",
                "nodeId": node.id,
                "rewardAddress": node.worker.reward_address,
                "balance": tokens(
                    self.chain.balance_of(node.worker.reward_address)
                ),
                "completedJobs": node.completed_jobs,
            }

    def node_snapshot(self, node_id: str) -> dict:
        with self._lock:
            self._expire_remote_nodes()
            node = self.nodes.get(node_id)
            if node is None:
                raise ValueError(f"unknown node: {node_id}")
            return node.snapshot(self.chain)

    @staticmethod
    def _completion(prompt: str, proof_root: str) -> str:
        normalized = prompt.lower()
        if "token" in normalized or "econom" in normalized:
            body = (
                "CTX is consumed for inference and paid to the workers that keep model "
                "shards resident and execute the route. Useful-work issuance is capped; "
                "transfers never create supply. Utility can create demand, but market "
                "price remains an external outcome."
            )
        elif any(word in normalized for word in ("scale", "ram", "parameter")):
            body = (
                "A model larger than one host is divided by transformer stage. Each "
                "activation moves to the next selected replica, while signed receipts "
                "bind the model version, weight commitment, input, output, and work."
            )
        else:
            body = (
                "The native execution route completed and every stage signature was "
                "verified before the control plane finalized its useful-work block. "
                "This devnet uses deterministic activation kernels; real tensor weights "
                "can replace that kernel without changing the chain contract."
            )
        return f"{body}\n\nBlock proof: {proof_root[:16]}…"

    def _execute_synthetic_route(
        self, inference_id: str, input_digest: str
    ) -> tuple[list[ExecutionReceipt], list[dict], str, dict]:
        activation_digest = input_digest
        receipts: list[ExecutionReceipt] = []
        route: list[dict] = []
        for shard in self.model["shards"]:
            node = self._route(shard["index"])
            stage_started = time.perf_counter()
            receipt = node.worker.execute(
                request_id=inference_id,
                model_id=self.model["id"],
                model_version=self.model["version"],
                shard_index=shard["index"],
                input_digest=activation_digest,
                weights_digest=shard["weightsDigest"],
                compute_units=shard["computeUnits"],
                challenge=secrets.token_hex(16),
            )
            self._verify_current_receipt(receipt, node, shard, activation_digest)
            stage_elapsed_ms = round((time.perf_counter() - stage_started) * 1000, 2)
            node.completed_jobs += 1
            receipts.append(receipt)
            activation_digest = receipt.output_digest
            route.append(
                {
                    "stage": shard["index"],
                    "nodeId": node.id,
                    "nodeName": node.name,
                    "region": node.region,
                    "backend": receipt.backend,
                    "kernelMicros": receipt.kernel_micros,
                    "controlPlaneMs": stage_elapsed_ms,
                    "networkLatencyMs": node.network_latency_ms,
                    "computeUnits": receipt.compute_units,
                    "receiptDigest": receipt_digest(receipt),
                }
            )
        return receipts, route, activation_digest, {
            "mode": "deterministic-kernel",
            "proofLevel": "signed-execution-receipts",
        }

    def _execute_real_route(
        self, inference_id: str, prompt: str, input_digest: str
    ) -> tuple[list[ExecutionReceipt], list[dict], str, dict]:
        if self.model_runtime is None:
            raise RuntimeError("real model runtime is unavailable")
        selected_nodes = [
            self._route(shard["index"]) for shard in self.model["shards"]
        ]
        if len({node.id for node in selected_nodes}) != len(selected_nodes):
            raise RuntimeError("distributed route requires distinct execution nodes")

        remote_jobs: list[tuple[dict, ComputeNode]] = []
        try:
            for shard, node in zip(self.model["shards"], selected_nodes):
                if not isinstance(node.worker, RemoteWorker):
                    continue
                node.worker.begin_job(
                    request_id=inference_id,
                    model_id=self.model["id"],
                    model_version=self.model["version"],
                    shard_index=shard["index"],
                    input_digest=input_digest,
                    weights_digest=shard["weightsDigest"],
                    compute_units=shard["computeUnits"],
                    challenge=secrets.token_hex(16),
                )
                remote_jobs.append((shard, node))
        except RuntimeError:
            for shard, node in remote_jobs:
                try:
                    node.worker.cancel_job(
                        request_id=inference_id,
                        model_id=self.model["id"],
                        model_version=self.model["version"],
                        shard_index=shard["index"],
                        input_digest=input_digest,
                        weights_digest=shard["weightsDigest"],
                        compute_units=shard["computeUnits"],
                        challenge=secrets.token_hex(16),
                    )
                except RuntimeError:
                    pass
            raise

        started = time.perf_counter()
        try:
            generation = self.model_runtime.generate(
                prompt=prompt,
                route=[node.id for node in selected_nodes],
            )
        except Exception:
            for shard, node in remote_jobs:
                try:
                    node.worker.cancel_job(
                        request_id=inference_id,
                        model_id=self.model["id"],
                        model_version=self.model["version"],
                        shard_index=shard["index"],
                        input_digest=input_digest,
                        weights_digest=shard["weightsDigest"],
                        compute_units=shard["computeUnits"],
                        challenge=secrets.token_hex(16),
                    )
                except RuntimeError:
                    pass
            raise
        runtime_elapsed_ms = (time.perf_counter() - started) * 1000
        completion_digest = sha256(generation["text"])
        activation_digest = input_digest
        receipts: list[ExecutionReceipt] = []
        route: list[dict] = []
        total_kernel_micros = int(
            (
                generation["timings"].get("prompt_ms", 0)
                + generation["timings"].get("predicted_ms", 0)
            )
            * 1_000
        )
        per_stage_micros = max(
            1, total_kernel_micros // len(self.model["shards"])
        )

        for index, (shard, node) in enumerate(
            zip(self.model["shards"], selected_nodes)
        ):
            is_last = index == len(self.model["shards"]) - 1
            output_digest = (
                completion_digest
                if is_last
                else sha256(
                    f"{inference_id}|{activation_digest}|{completion_digest}|"
                    f"rpc-boundary|{index}"
                )
            )
            receipt = node.worker.attest(
                request_id=inference_id,
                model_id=self.model["id"],
                model_version=self.model["version"],
                shard_index=shard["index"],
                input_digest=activation_digest,
                output_digest=output_digest,
                weights_digest=shard["weightsDigest"],
                compute_units=shard["computeUnits"],
                challenge=secrets.token_hex(16),
                backend="llama.cpp-rpc",
                kernel_micros=per_stage_micros,
            )
            self._verify_current_receipt(receipt, node, shard, activation_digest)
            node.completed_jobs += 1
            receipts.append(receipt)
            activation_digest = output_digest
            route.append(
                {
                    "stage": shard["index"],
                    "layers": shard["layerRange"],
                    "nodeId": node.id,
                    "nodeName": node.name,
                    "region": node.region,
                    "backend": receipt.backend,
                    "kernelMicros": receipt.kernel_micros,
                    "controlPlaneMs": round(runtime_elapsed_ms / len(selected_nodes), 2),
                    "networkLatencyMs": node.network_latency_ms,
                    "computeUnits": receipt.compute_units,
                    "receiptDigest": receipt_digest(receipt),
                }
            )

        return receipts, route, activation_digest, {
            "mode": "real-transformer",
            "proofLevel": "signed-participation-attestation",
            "text": generation["text"],
            "usage": generation["usage"],
            "timings": generation["timings"],
            "runtimeModel": generation["runtimeModel"],
            "systemFingerprint": generation["systemFingerprint"],
        }

    @staticmethod
    def _fallback_token_count(text: str) -> int:
        return max(1, math.ceil(len(text.encode("utf-8")) / 4))

    def _ready_capacity_units(self) -> tuple[int, int]:
        self._expire_remote_nodes()
        ready = sum(
            int(shard["computeUnits"])
            for node in self.nodes.values()
            if node.status == "online"
            for shard in node.shards
            if shard["modelId"] == self.model["id"]
        )
        target = (
            sum(int(shard["computeUnits"]) for shard in self.model["shards"])
            * CAPACITY_TARGET_REPLICAS
        )
        return min(ready, target), target

    def run_inference(self, prompt: str, model_id: str | None = None) -> dict:
        if not isinstance(prompt, str) or len(prompt.strip()) < 2:
            raise ValueError("prompt must contain at least 2 characters")
        if len(prompt) > 8_000:
            raise ValueError("prompt exceeds the 8,000 character devnet limit")
        if model_id not in (None, self.model["id"]):
            raise ValueError(f"unknown model: {model_id}")

        with self._lock:
            inference_id = f"inf_{secrets.token_hex(8)}"
            started = time.perf_counter()
            input_digest = sha256(
                f"{inference_id}|{self.model['id']}|{prompt.strip()}"
            )
            prompt_digest = sha256(prompt.strip())
            if self.model_runtime is None:
                receipts, route, activation_digest, runtime = (
                    self._execute_synthetic_route(inference_id, input_digest)
                )
            else:
                receipts, route, activation_digest, runtime = (
                    self._execute_real_route(
                        inference_id, prompt.strip(), input_digest
                    )
                )

            proof = {
                "algorithm": "ed25519-receipt-merkle-v1",
                "root": merkle_root([receipt_digest(receipt) for receipt in receipts]),
                "receipts": [receipt.to_dict() for receipt in receipts],
            }
            proof_check = verify_proof_bundle(proof)
            if not proof_check["valid"]:
                raise RuntimeError("final proof bundle did not verify")

            output_text = runtime.get("text") or self._completion(
                prompt.strip(), proof["root"]
            )
            usage = runtime.get("usage", {})
            input_tokens = int(
                usage.get("prompt_tokens")
                or self._fallback_token_count(prompt.strip())
            )
            output_tokens = int(
                usage.get("completion_tokens")
                or self._fallback_token_count(output_text)
            )
            provider_compute_units: dict[str, int] = {}
            provider_rates_micro: dict[str, int] = {}
            for receipt, route_stage in zip(receipts, route):
                address = address_from_public_key(receipt.public_key)
                provider_compute_units[address] = (
                    provider_compute_units.get(address, 0)
                    + int(receipt.compute_units)
                )
                node = self.nodes[route_stage["nodeId"]]
                existing_rate = provider_rates_micro.setdefault(
                    address, node.price_micro_per_work_unit
                )
                if existing_rate != node.price_micro_per_work_unit:
                    raise RuntimeError(
                        "one provider identity cannot use multiple prices in a route"
                    )
            ready_capacity_units, target_capacity_units = (
                self._ready_capacity_units()
            )
            economic_context = self.chain.economic_context()
            economics = build_economics_record(
                inference_index=economic_context["inferenceIndex"],
                model_compute_units=sum(
                    int(receipt.compute_units) for receipt in receipts
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                kernel_micros=sum(
                    int(receipt.kernel_micros) for receipt in receipts
                ),
                provider_rates_micro=provider_rates_micro,
                provider_compute_units=provider_compute_units,
                ready_capacity_units=ready_capacity_units,
                target_capacity_units=target_capacity_units,
                epoch_fees_before_micro=economic_context["epochFeesMicro"],
                epoch_issuance_before_micro=economic_context[
                    "epochIssuanceMicro"
                ],
            )
            block = self.chain.commit_inference(
                inference_id=inference_id,
                payer=self.demo_wallet.address,
                model=self.model,
                proof=proof,
                economics=economics,
                input_digest=input_digest,
                output_digest=activation_digest,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            reward_transactions = [
                transaction
                for transaction in block["transactions"]
                if transaction["type"] == "provider_reward"
            ]
            inference_fee_transaction = next(
                transaction
                for transaction in block["transactions"]
                if transaction["type"] == "inference_fee"
            )
            accounting = {
                "symbol": TOKEN_SYMBOL,
                "economicsVersion": economics["version"],
                "endUserFee": 0,
                "sponsorFee": tokens(
                    inference_fee_transaction["amountMicro"]
                ),
                "fee": tokens(inference_fee_transaction["amountMicro"]),
                "burned": tokens(
                    sum(
                        transaction["amountMicro"]
                        for transaction in block["transactions"]
                        if transaction["type"] == "burn"
                    )
                ),
                "protocolFee": tokens(
                    sum(
                        transaction["amountMicro"]
                        for transaction in block["transactions"]
                        if transaction["type"] == "protocol_fee"
                    )
                ),
                "providerFeeRewards": tokens(
                    sum(
                        transaction["feeAmountMicro"]
                        for transaction in reward_transactions
                    )
                ),
                "providerRewards": tokens(
                    sum(
                        transaction["feeAmountMicro"]
                        + transaction["issuanceAmountMicro"]
                        for transaction in reward_transactions
                    )
                ),
                "quotedProviderCost": tokens(
                    economics["pricing"]["quotedProviderCostMicro"]
                ),
                "issuance": tokens(
                    sum(
                        transaction["issuanceAmountMicro"]
                        for transaction in reward_transactions
                    )
                ),
                "epoch": economics["emission"]["epoch"],
                "epochBudget": tokens(
                    economics["emission"]["epochBudgetMicro"]
                ),
                "capacityUnlockPercent": (
                    economics["emission"]["capacityUnlockBps"] / 100
                ),
                "demandUnlockPercent": (
                    economics["emission"]["demandUnlockBps"] / 100
                ),
            }
            rewards_by_address = {
                transaction["to"]: (
                    transaction["feeAmountMicro"]
                    + transaction["issuanceAmountMicro"]
                )
                for transaction in reward_transactions
            }
            for shard, receipt, route_stage in zip(
                self.model["shards"], receipts, route
            ):
                node = self.nodes[route_stage["nodeId"]]
                if not isinstance(node.worker, RemoteWorker):
                    continue
                reward_micro = rewards_by_address.get(
                    node.worker.reward_address, 0
                )
                try:
                    node.worker.settle_job(
                        request_id=inference_id,
                        model_id=self.model["id"],
                        model_version=self.model["version"],
                        shard_index=shard["index"],
                        input_digest=receipt.input_digest,
                        weights_digest=shard["weightsDigest"],
                        compute_units=shard["computeUnits"],
                        challenge=secrets.token_hex(16),
                        blockHeight=block["height"],
                        blockHash=block["hash"],
                        proofRoot=proof["root"],
                        rewardMicro=reward_micro,
                        reward=tokens(reward_micro),
                        providerBalance=tokens(
                            self.chain.balance_of(node.worker.reward_address)
                        ),
                        completedJobs=node.completed_jobs,
                    )
                except RuntimeError as error:
                    self._event(
                        "warning",
                        f"{node.name} settlement notice failed",
                        str(error),
                        nodeId=node.id,
                        inferenceId=inference_id,
                    )
            result = {
                "id": inference_id,
                "status": "finalized",
                "simulation": self.model_runtime is None,
                "access": {
                    "mode": "network-sponsored",
                    "endUserWalletRequired": False,
                    "endUserFee": 0,
                    "symbol": TOKEN_SYMBOL,
                },
                "model": {
                    key: self.model[key]
                    for key in ("id", "name", "version", "shardCount")
                },
                "input": {
                    "prompt": prompt.strip(),
                    "promptDigest": prompt_digest,
                },
                "output": {
                    "text": output_text,
                    "outputDigest": activation_digest,
                },
                "performance": {
                    "wallTimeMs": elapsed_ms,
                    "stages": len(route),
                    "kernelMicros": sum(item["kernelMicros"] for item in route),
                    "usefulWorkUnits": sum(
                        receipt.compute_units for receipt in receipts
                    ),
                    "tokens": runtime.get("usage", {}),
                    "timings": runtime.get("timings", {}),
                },
                "runtime": {
                    key: value
                    for key, value in runtime.items()
                    if key not in {"text", "usage", "timings"}
                },
                "route": route,
                "proof": proof,
                "block": Blockchain.public_block(block),
                "accounting": accounting,
            }
            self.recent_inferences.insert(
                0,
                {
                    "id": inference_id,
                    "prompt": prompt.strip(),
                    "model": self.model["name"],
                    "blockHeight": block["height"],
                    "blockHash": block["hash"],
                    "proofRoot": proof["root"],
                    "wallTimeMs": elapsed_ms,
                    "completedAt": utc_now(),
                },
            )
            del self.recent_inferences[20:]
            self._event(
                "block",
                f"Block #{block['height']} finalized",
                f"{len(receipts)} receipts · {proof['root'][:12]} · "
                f"{accounting['providerRewards']} {TOKEN_SYMBOL} rewarded",
                inferenceId=inference_id,
                blockHash=block["hash"],
            )
            return result

    def set_node_status(self, node_id: str, status: str) -> dict:
        if status not in {"online", "offline"}:
            raise ValueError("node status must be online or offline")
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise ValueError(f"unknown node: {node_id}")
            if node.remote:
                raise ValueError(
                    "remote node status is controlled by signed readiness and heartbeats"
                )
            node.status = status
            self._event(
                "node" if status == "online" else "warning",
                f"{node.name} is {status}",
                f"{node.region} · {len(node.shards)} resident shards",
                nodeId=node.id,
            )
            return node.snapshot(self.chain)

    def public_model(self) -> dict:
        placements = []
        for shard in self.model["shards"]:
            placements.append(
                {
                    **shard,
                    "replicas": [
                        node.id
                        for node in self.nodes.values()
                        if node.hosts(self.model["id"], shard["index"])
                    ],
                }
            )
        return {
            **{
                key: value
                for key, value in self.model.items()
                if key != "shards"
            },
            "modelMemoryGiB": self.model["shardCount"]
            * self.model["shardSizeGiB"],
            "replicatedMemoryGiB": round(
                sum(node.allocated_memory_gib for node in self.nodes.values()), 6
            ),
            "placements": placements,
        }

    def snapshot(self) -> dict:
        with self._lock:
            self._expire_remote_nodes()
            node_snapshots = [
                node.snapshot(self.chain) for node in self.nodes.values()
            ]
            online = [node for node in node_snapshots if node["status"] == "online"]
            inference_blocks = [
                block for block in self.chain.blocks if block["kind"] == "inference"
            ]
            return {
                "name": "CortexGrid",
                "environment": "devnet",
                "protocolVersion": "0.1.0",
                "coordinatorPublicKey": self.coordinator_wallet.public_key_hex,
                "status": (
                    "healthy"
                    if len(online) == len(node_snapshots)
                    else "degraded"
                ),
                "metrics": {
                    "nodesOnline": len(online),
                    "nodesTotal": len(node_snapshots),
                    "totalMemoryGiB": sum(
                        node["memoryGiB"] for node in node_snapshots
                    ),
                    "allocatedMemoryGiB": sum(
                        node["allocatedMemoryGiB"] for node in node_snapshots
                    ),
                    "modelsReady": 1,
                    "inferencesFinalized": len(inference_blocks),
                    "verifiedReceipts": sum(
                        len(block["payload"]["proof"]["receipts"])
                        for block in inference_blocks
                    ),
                    "chainHeight": len(self.chain.blocks) - 1,
                },
                "model": self.public_model(),
                "modelRuntime": (
                    self.model_runtime.snapshot()
                    if self.model_runtime is not None
                    else {
                        "available": False,
                        "active": False,
                        "engine": "cortex-deterministic-kernel",
                    }
                ),
                "nodes": node_snapshots,
                "chain": self.chain.snapshot(),
                "recentEvents": list(self.events[:30]),
                "recentInferences": list(self.recent_inferences),
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.model_runtime is not None:
            self.model_runtime.close()
        for node in self.nodes.values():
            node.worker.close()
