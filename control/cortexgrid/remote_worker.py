from __future__ import annotations

from .executor import ExecutionReceipt, verify_receipt_signature
from .join_protocol import (
    JOIN_PROTOCOL,
    WORK_TICKET_DOMAIN,
    request_json,
    sign_payload,
)
from .wallet import Wallet, address_from_public_key


class RemoteWorker:
    """Executor-compatible client for a provider-owned node daemon."""

    def __init__(
        self,
        *,
        node_id: str,
        public_key: str,
        capabilities: list[str],
        endpoint: str,
        assignment_id: str,
        coordinator_wallet: Wallet,
    ):
        self.node_id = node_id
        self.public_key = public_key
        self.capabilities = list(capabilities)
        self.endpoint = endpoint.rstrip("/")
        self.assignment_id = assignment_id
        self.coordinator_wallet = coordinator_wallet

    @property
    def reward_address(self) -> str:
        return address_from_public_key(self.public_key)

    def _submit(self, operation: str, ticket: dict) -> ExecutionReceipt:
        response = request_json(
            f"{self.endpoint}/v1/{operation}",
            payload=self._envelope(ticket),
            timeout=120,
        )
        receipt = ExecutionReceipt.from_dict(response["receipt"])
        expected = {
            "node_id": self.node_id,
            "public_key": self.public_key,
            "request_id": ticket["requestId"],
            "model_id": ticket["modelId"],
            "model_version": ticket["modelVersion"],
            "shard_index": int(ticket["shardIndex"]),
            "input_digest": ticket["inputDigest"],
            "weights_digest": ticket["weightsDigest"],
            "compute_units": int(ticket["computeUnits"]),
            "challenge": ticket["challenge"],
        }
        if any(getattr(receipt, key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"remote node {self.node_id} returned a mismatched receipt"
            )
        if operation == "attest" and (
            receipt.output_digest != ticket["outputDigest"]
            or receipt.backend != ticket["backend"]
            or receipt.kernel_micros != int(ticket["kernelMicros"])
        ):
            raise RuntimeError(
                f"remote node {self.node_id} changed its attestation ticket"
            )
        if not verify_receipt_signature(receipt):
            raise RuntimeError(f"remote node {self.node_id} returned an invalid receipt")
        return receipt

    def _envelope(self, ticket: dict) -> dict:
        return {
            "ticket": ticket,
            "coordinatorPublicKey": self.coordinator_wallet.public_key_hex,
            "coordinatorSignature": sign_payload(
                self.coordinator_wallet, WORK_TICKET_DOMAIN, ticket
            ),
        }

    def _job_ticket(
        self,
        *,
        operation: str,
        request_id: str,
        model_id: str,
        model_version: str,
        shard_index: int,
        input_digest: str,
        weights_digest: str,
        compute_units: int,
        challenge: str,
        **metadata: object,
    ) -> dict:
        return {
            "protocol": JOIN_PROTOCOL,
            "operation": operation,
            "assignmentId": self.assignment_id,
            "nodeId": self.node_id,
            "requestId": request_id,
            "modelId": model_id,
            "modelVersion": model_version,
            "shardIndex": shard_index,
            "inputDigest": input_digest,
            "weightsDigest": weights_digest,
            "computeUnits": compute_units,
            "challenge": challenge,
            **metadata,
        }

    def begin_job(self, **values: object) -> dict:
        ticket = self._job_ticket(operation="start", **values)
        return request_json(
            f"{self.endpoint}/v1/jobs/start",
            payload=self._envelope(ticket),
        )

    def cancel_job(self, **values: object) -> dict:
        ticket = self._job_ticket(operation="cancel", **values)
        return request_json(
            f"{self.endpoint}/v1/jobs/cancel",
            payload=self._envelope(ticket),
        )

    def settle_job(self, **values: object) -> dict:
        ticket = self._job_ticket(operation="settle", **values)
        return request_json(
            f"{self.endpoint}/v1/jobs/settle",
            payload=self._envelope(ticket),
        )

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
        ticket = {
            "protocol": JOIN_PROTOCOL,
            "operation": "execute",
            "assignmentId": self.assignment_id,
            "nodeId": self.node_id,
            "requestId": request_id,
            "modelId": model_id,
            "modelVersion": model_version,
            "shardIndex": shard_index,
            "inputDigest": input_digest,
            "weightsDigest": weights_digest,
            "computeUnits": compute_units,
            "challenge": challenge,
        }
        return self._submit("execute", ticket)

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
        ticket = {
            "protocol": JOIN_PROTOCOL,
            "operation": "attest",
            "assignmentId": self.assignment_id,
            "nodeId": self.node_id,
            "requestId": request_id,
            "modelId": model_id,
            "modelVersion": model_version,
            "shardIndex": shard_index,
            "inputDigest": input_digest,
            "outputDigest": output_digest,
            "weightsDigest": weights_digest,
            "computeUnits": compute_units,
            "challenge": challenge,
            "backend": backend,
            "kernelMicros": kernel_micros,
        }
        return self._submit("attest", ticket)

    def close(self) -> None:
        return
