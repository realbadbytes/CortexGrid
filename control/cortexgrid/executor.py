from __future__ import annotations

import os
import selectors
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .wallet import Wallet, address_from_public_key


PROTOCOL = "cg-executor-v1"


@dataclass(frozen=True)
class ExecutionReceipt:
    request_id: str
    model_id: str
    model_version: str
    shard_index: int
    node_id: str
    input_digest: str
    output_digest: str
    weights_digest: str
    compute_units: int
    challenge: str
    backend: str
    kernel_micros: int
    public_key: str
    signature: str

    def canonical_message(self) -> str:
        return "|".join(
            [
                self.request_id,
                self.model_id,
                self.model_version,
                str(self.shard_index),
                self.node_id,
                self.input_digest,
                self.output_digest,
                self.weights_digest,
                str(self.compute_units),
                self.challenge,
                self.backend,
                str(self.kernel_micros),
            ]
        )

    def to_dict(self) -> dict:
        raw = asdict(self)
        return {
            "".join(
                [part if index == 0 else part.title() for index, part in enumerate(key.split("_"))]
            ): value
            for key, value in raw.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionReceipt":
        return cls(
            request_id=data["requestId"],
            model_id=data["modelId"],
            model_version=data["modelVersion"],
            shard_index=int(data["shardIndex"]),
            node_id=data["nodeId"],
            input_digest=data["inputDigest"],
            output_digest=data["outputDigest"],
            weights_digest=data["weightsDigest"],
            compute_units=int(data["computeUnits"]),
            challenge=data["challenge"],
            backend=data["backend"],
            kernel_micros=int(data["kernelMicros"]),
            public_key=data["publicKey"],
            signature=data["signature"],
        )


def verify_receipt_signature(receipt: ExecutionReceipt) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(receipt.public_key))
        key.verify(
            bytes.fromhex(receipt.signature),
            receipt.canonical_message().encode("utf-8"),
        )
        return True
    except Exception:
        return False


class NativeWorker:
    """One persistent C++ executor process with a tiny tab-delimited protocol."""

    def __init__(self, node_id: str, executable: Path, timeout_seconds: float = 10):
        self.node_id = node_id
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.wallet = Wallet.generate()
        self.public_key = ""
        self.capabilities: list[str] = []
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._start()

    @property
    def reward_address(self) -> str:
        return address_from_public_key(self.public_key)

    def _start(self) -> None:
        if not self.executable.exists():
            raise RuntimeError(
                f"Native executor not found at {self.executable}. Run `make build` first."
            )
        environment = os.environ.copy()
        environment["CORTEX_NODE_PRIVATE_KEY"] = self.wallet.private_key_hex
        self._process = subprocess.Popen(
            [str(self.executable), "--serve", "--node-id", self.node_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        ready = self._readline()
        fields = ready.rstrip("\n").split("\t")
        if len(fields) != 4 or fields[0] != "READY" or fields[1] != PROTOCOL:
            self.close()
            raise RuntimeError(f"Native executor returned an invalid handshake: {ready!r}")
        self.capabilities = fields[2].split(",")
        self.public_key = fields[3]
        if self.public_key != self.wallet.public_key_hex:
            self.close()
            raise RuntimeError("Native executor public key does not match its identity")

    def _readline(self) -> str:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Native executor is not running")
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(self.timeout_seconds):
                raise TimeoutError(f"Native executor {self.node_id} timed out")
            line = self._process.stdout.readline()
        finally:
            selector.close()
        if not line:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().strip()
            raise RuntimeError(
                f"Native executor {self.node_id} exited unexpectedly: {stderr}"
            )
        return line

    def execute(
        self,
        *,
        request_id: str,
        model_id: str,
        model_version: str,
        shard_index: int,
        input_digest: str,
        weights_digest: str,
        compute_units: int,
        challenge: str,
    ) -> ExecutionReceipt:
        values = [
            request_id,
            model_id,
            model_version,
            str(shard_index),
            input_digest,
            weights_digest,
            str(compute_units),
            challenge,
        ]
        if any("\t" in value or "\n" in value for value in values):
            raise ValueError("Executor protocol fields cannot contain tabs or newlines")

        with self._lock:
            if (
                self._process is None
                or self._process.poll() is not None
                or self._process.stdin is None
            ):
                raise RuntimeError(f"Native executor {self.node_id} is unavailable")
            self._process.stdin.write("EXEC\t" + "\t".join(values) + "\n")
            self._process.stdin.flush()
            response = self._readline().rstrip("\n").split("\t")

        if response[0] == "ERROR":
            raise RuntimeError(response[1] if len(response) > 1 else "executor error")
        if len(response) != 6 or response[0] != "RESULT":
            raise RuntimeError(f"Invalid native executor response: {response!r}")
        if response[1] != request_id:
            raise RuntimeError("Native executor response belongs to another request")

        return ExecutionReceipt(
            request_id=request_id,
            model_id=model_id,
            model_version=model_version,
            shard_index=shard_index,
            node_id=self.node_id,
            input_digest=input_digest,
            output_digest=response[2],
            weights_digest=weights_digest,
            compute_units=compute_units,
            challenge=challenge,
            backend=response[3],
            kernel_micros=int(response[4]),
            public_key=self.public_key,
            signature=response[5],
        )

    def attest(
        self,
        *,
        request_id: str,
        model_id: str,
        model_version: str,
        shard_index: int,
        input_digest: str,
        output_digest: str,
        weights_digest: str,
        compute_units: int,
        challenge: str,
        backend: str,
        kernel_micros: int,
    ) -> ExecutionReceipt:
        """Sign a receipt for work performed by this node's model RPC backend."""
        values = [
            request_id,
            model_id,
            model_version,
            str(shard_index),
            input_digest,
            output_digest,
            weights_digest,
            str(compute_units),
            challenge,
            backend,
            str(kernel_micros),
        ]
        if any("\t" in value or "\n" in value for value in values):
            raise ValueError("Executor protocol fields cannot contain tabs or newlines")

        with self._lock:
            if (
                self._process is None
                or self._process.poll() is not None
                or self._process.stdin is None
            ):
                raise RuntimeError(f"Native executor {self.node_id} is unavailable")
            self._process.stdin.write("ATTEST\t" + "\t".join(values) + "\n")
            self._process.stdin.flush()
            response = self._readline().rstrip("\n").split("\t")

        if response[0] == "ERROR":
            raise RuntimeError(response[1] if len(response) > 1 else "executor error")
        if len(response) != 3 or response[0] != "ATTESTED":
            raise RuntimeError(f"Invalid native attestation response: {response!r}")
        if response[1] != request_id:
            raise RuntimeError("Native attestation belongs to another request")

        receipt = ExecutionReceipt(
            request_id=request_id,
            model_id=model_id,
            model_version=model_version,
            shard_index=shard_index,
            node_id=self.node_id,
            input_digest=input_digest,
            output_digest=output_digest,
            weights_digest=weights_digest,
            compute_units=compute_units,
            challenge=challenge,
            backend=backend,
            kernel_micros=kernel_micros,
            public_key=self.public_key,
            signature=response[2],
        )
        if not verify_receipt_signature(receipt):
            raise RuntimeError(f"Native executor {self.node_id} signed an invalid receipt")
        return receipt

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write("STOP\n")
                process.stdin.flush()
                process.wait(timeout=1)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
