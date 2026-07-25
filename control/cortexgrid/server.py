from __future__ import annotations

import json
import mimetypes
import os
import re
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .blockchain import Blockchain, verify_proof_bundle
from .network import CortexNetwork


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = PROJECT_ROOT / "public"
MAX_BODY_BYTES = 1_000_000


class ApiHandler(BaseHTTPRequestHandler):
    server: "CortexHttpServer"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[http] {self.address_string()} {format % args}")

    def _json(self, status: int, body: object) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            raise ValueError("request body must be valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _static(self, path: str) -> bool:
        requested = "index.html" if path == "/" else path.lstrip("/")
        file_path = (PUBLIC_DIR / requested).resolve()
        if PUBLIC_DIR.resolve() not in file_path.parents and file_path != PUBLIC_DIR:
            return False
        if not file_path.is_file():
            return False
        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)
        return True

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        network = self.server.network
        try:
            if path == "/api/health":
                self._json(
                    200,
                    {
                        "ok": True,
                        "service": "cortex-grid-control-plane",
                        "executionPlane": "c++/cuda",
                        "chainValid": network.chain.validate()["valid"],
                    },
                )
            elif path == "/api/network":
                self._json(200, network.snapshot())
            elif path == "/api/chain":
                self._json(200, network.chain.snapshot())
            elif path == "/api/nodes":
                self._json(
                    200,
                    {
                        "nodes": [
                            node.snapshot(network.chain)
                            for node in network.nodes.values()
                        ]
                    },
                )
            elif path == "/api/models":
                self._json(200, {"models": [network.public_model()]})
            elif path == "/api/openapi.json":
                self._json(200, openapi_document(self.server.server_port))
            elif match := re.fullmatch(r"/api/blocks/(\d+)", path):
                height = int(match.group(1))
                if height >= len(network.chain.blocks):
                    self._json(404, {"error": "block_not_found"})
                else:
                    self._json(
                        200,
                        Blockchain.public_block(
                            network.chain.blocks[height], include_receipts=True
                        ),
                    )
            elif path.startswith("/api/"):
                self._json(404, {"error": "not_found", "path": path})
            elif not self._static(path):
                self._json(404, {"error": "not_found", "path": path})
        except Exception as error:
            self._json(500, {"error": "internal_error", "message": str(error)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        network = self.server.network
        try:
            body = self._body()
            if path == "/api/inference":
                result = network.run_inference(
                    body.get("prompt", ""), body.get("modelId")
                )
                self._json(201, result)
            elif path == "/api/proofs/verify":
                self._json(200, verify_proof_bundle(body.get("proof", body)))
            elif path == "/api/transfers":
                block = network.chain.commit_transfer(body)
                self._json(201, Blockchain.public_block(block))
            elif match := re.fullmatch(
                r"/api/nodes/([a-z0-9_-]+)/status", path
            ):
                self._json(
                    200, network.set_node_status(match.group(1), body.get("status", ""))
                )
            else:
                self._json(404, {"error": "not_found", "path": path})
        except (KeyError, TypeError, ValueError) as error:
            self._json(400, {"error": "invalid_request", "message": str(error)})
        except RuntimeError as error:
            self._json(503, {"error": "network_unavailable", "message": str(error)})
        except Exception as error:
            self._json(500, {"error": "internal_error", "message": str(error)})


class CortexHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], network: CortexNetwork):
        self.network = network
        super().__init__(address, ApiHandler)

    def server_close(self) -> None:
        self.network.close()
        super().server_close()


def openapi_document(port: int) -> dict:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "CortexGrid API",
            "version": "0.1.0",
            "description": (
                "Python control plane for native distributed execution and the "
                "proof-of-useful-inference CTX chain."
            ),
        },
        "servers": [{"url": f"http://localhost:{port}"}],
        "paths": {
            "/api/network": {"get": {"summary": "Network-wide snapshot"}},
            "/api/chain": {"get": {"summary": "CTX chain and balances"}},
            "/api/blocks/{height}": {
                "get": {"summary": "Full block with inference receipts"}
            },
            "/api/inference": {
                "post": {
                    "summary": "Execute inference and finalize its useful-work block"
                }
            },
            "/api/proofs/verify": {
                "post": {"summary": "Verify a Merkle bundle of Ed25519 receipts"}
            },
            "/api/transfers": {
                "post": {"summary": "Submit an Ed25519-signed CTX transfer"}
            },
            "/api/nodes/{nodeId}/status": {
                "post": {"summary": "Toggle a devnet node to exercise failover"}
            },
        },
    }


def main() -> None:
    port = int(os.environ.get("PORT", "4173"))
    network = CortexNetwork()
    server: CortexHttpServer | None = None

    def stop(_signal: int, _frame: object) -> None:
        raise KeyboardInterrupt

    try:
        server = CortexHttpServer(("0.0.0.0", port), network)
        signal.signal(signal.SIGTERM, stop)
        print(f"CortexGrid devnet listening on http://localhost:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping CortexGrid devnet")
    finally:
        if server is None:
            network.close()
        else:
            server.server_close()


if __name__ == "__main__":
    main()
