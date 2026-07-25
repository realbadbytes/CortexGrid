from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class DistributedModelRuntime:
    """Owns private llama.cpp RPC participants and one generation gateway."""

    def __init__(
        self,
        *,
        model_path: Path,
        rpc_binary: Path,
        server_binary: Path,
        node_ids: list[str],
        data_dir: Path,
        base_port: int | None = None,
    ):
        self.model_path = model_path
        self.rpc_binary = rpc_binary
        self.server_binary = server_binary
        self.node_ids = list(node_ids)
        self.data_dir = data_dir
        configured_base = os.environ.get("CORTEX_MODEL_PORT_BASE")
        self.base_port = (
            base_port
            if base_port is not None
            else int(configured_base)
            if configured_base is not None
            else self._find_available_port_block(len(self.node_ids))
        )
        self.gateway_port = self.base_port + 80
        self._node_ports = {
            node_id: self.base_port + index
            for index, node_id in enumerate(self.node_ids)
        }
        self._rpc_processes: dict[str, subprocess.Popen[bytes]] = {}
        self._rpc_logs: dict[str, object] = {}
        self._gateway: subprocess.Popen[bytes] | None = None
        self._gateway_log: object | None = None
        self._active_route: tuple[str, ...] = ()
        self._closed = False
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _find_available_port_block(node_count: int) -> int:
        """Choose a compact local port block for one isolated model runtime."""
        for base_port in range(53100, 60000, 100):
            candidates = [base_port + index for index in range(node_count)]
            candidates.append(base_port + 80)
            reservations: list[socket.socket] = []
            try:
                for port in candidates:
                    reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    reservations.append(reservation)
                    reservation.bind(("127.0.0.1", port))
                return base_port
            except OSError:
                continue
            finally:
                for reservation in reservations:
                    reservation.close()
        raise RuntimeError("no private port block is available for the model runtime")

    @classmethod
    def is_available(
        cls, model_path: Path, rpc_binary: Path, server_binary: Path
    ) -> bool:
        return all(path.is_file() for path in (model_path, rpc_binary, server_binary))

    def _wait_for_port(self, port: int, process: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"model runtime process exited with status {process.returncode}"
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError(f"model runtime did not bind port {port}")

    def _start_rpc(self, node_id: str) -> None:
        current = self._rpc_processes.get(node_id)
        if current is not None and current.poll() is None:
            return
        port = self._node_ports[node_id]
        log = (self.data_dir / f"rpc-{node_id}.log").open("ab")
        process = subprocess.Popen(
            [
                str(self.rpc_binary),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--threads",
                "2",
                "--cache",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self._rpc_processes[node_id] = process
        self._rpc_logs[node_id] = log
        self._wait_for_port(port, process)

    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.gateway_port}/health", timeout=0.5
            ) as response:
                body = json.loads(response.read())
                return response.status == 200 and body.get("status") == "ok"
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return False

    def _stop_gateway(self) -> None:
        process = self._gateway
        self._gateway = None
        self._active_route = ()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._gateway_log is not None:
            self._gateway_log.close()
            self._gateway_log = None

    def ensure_route(self, node_ids: list[str]) -> None:
        route = tuple(node_ids)
        if not route or len(set(route)) != len(route):
            raise ValueError("a model route needs one or more distinct nodes")
        if any(node_id not in self._node_ports for node_id in route):
            raise ValueError("route contains an unknown model participant")
        if (
            route == self._active_route
            and self._gateway is not None
            and self._gateway.poll() is None
            and self._health()
        ):
            return

        self._stop_gateway()
        for node_id in route:
            self._start_rpc(node_id)
        endpoints = ",".join(
            f"127.0.0.1:{self._node_ports[node_id]}" for node_id in route
        )
        tensor_split = ",".join("1" for _ in route)
        self._gateway_log = (self.data_dir / "model-gateway.log").open("ab")
        self._gateway = subprocess.Popen(
            [
                str(self.server_binary),
                "--model",
                str(self.model_path),
                "--alias",
                "cortex-smollm2",
                "--rpc",
                endpoints,
                "--gpu-layers",
                "all",
                "--split-mode",
                "layer",
                "--tensor-split",
                tensor_split,
                "--ctx-size",
                "1024",
                "--parallel",
                "1",
                "--threads",
                "2",
                "--threads-batch",
                "2",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.gateway_port),
                "--no-webui",
                "--metrics",
            ],
            stdout=self._gateway_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._gateway.poll() is not None:
                raise RuntimeError(
                    f"model gateway exited with status {self._gateway.returncode}"
                )
            if self._health():
                self._active_route = route
                return
            time.sleep(0.2)
        self._stop_gateway()
        raise TimeoutError("distributed model gateway did not become healthy")

    def generate(
        self, *, prompt: str, route: list[str], max_tokens: int = 48
    ) -> dict:
        self.ensure_route(route)
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.gateway_port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "cortex-smollm2",
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "seed": 7,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"model gateway rejected inference: {detail}") from error
        completion = body["choices"][0]["message"]["content"].strip()
        if not completion:
            raise RuntimeError("distributed model returned an empty completion")
        return {
            "text": completion,
            "usage": body.get("usage", {}),
            "timings": body.get("timings", {}),
            "runtimeModel": body.get("model", "cortex-smollm2"),
            "systemFingerprint": body.get("system_fingerprint"),
            "route": list(self._active_route),
        }

    def snapshot(self) -> dict:
        return {
            "available": True,
            "active": self._health() if self._gateway is not None else False,
            "engine": "llama.cpp",
            "transport": "private-rpc-tcp",
            "modelFile": self.model_path.name,
            "activeRoute": list(self._active_route),
            "participants": [
                {
                    "nodeId": node_id,
                    "endpoint": f"127.0.0.1:{self._node_ports[node_id]}",
                    "resident": (
                        node_id in self._rpc_processes
                        and self._rpc_processes[node_id].poll() is None
                    ),
                }
                for node_id in self.node_ids
            ],
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_gateway()
        for process in self._rpc_processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self._rpc_processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        for log in self._rpc_logs.values():
            log.close()
        self._rpc_processes.clear()
        self._rpc_logs.clear()
