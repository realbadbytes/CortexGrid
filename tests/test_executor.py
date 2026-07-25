from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
from pathlib import Path

from cortexgrid.executor import NativeWorker, verify_receipt_signature
from cortexgrid.util import sha256


EXECUTOR_PATH = Path(
    os.environ.get(
        "CORTEX_EXECUTOR_PATH",
        Path(__file__).resolve().parents[1] / "build" / "cortex_worker",
    )
)


class NativeWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = NativeWorker("test-node", EXECUTOR_PATH)

    def tearDown(self) -> None:
        self.worker.close()

    def execute(self):
        return self.worker.execute(
            request_id="inf_test",
            model_id="model",
            model_version="1",
            shard_index=0,
            input_digest=sha256("activation"),
            weights_digest=sha256("weights"),
            compute_units=2,
            challenge="0123456789abcdef",
        )

    def test_native_receipt_has_publicly_verifiable_signature(self) -> None:
        receipt = self.execute()
        self.assertEqual(receipt.backend, "cpu")
        self.assertEqual(len(receipt.output_digest), 64)
        self.assertEqual(len(receipt.signature), 128)
        self.assertTrue(verify_receipt_signature(receipt))

    def test_tampered_receipt_is_rejected(self) -> None:
        receipt = self.execute()
        tampered = dataclasses.replace(receipt, output_digest=sha256("tampered"))
        self.assertFalse(verify_receipt_signature(tampered))

    def test_same_inputs_are_deterministic(self) -> None:
        first = self.execute()
        second = self.execute()
        self.assertEqual(first.output_digest, second.output_digest)

    def test_worker_can_attest_its_rpc_model_participation(self) -> None:
        receipt = self.worker.attest(
            request_id="inf_rpc",
            model_id="smollm",
            model_version="q4",
            shard_index=1,
            input_digest=sha256("input"),
            output_digest=sha256("output"),
            weights_digest=sha256("weights"),
            compute_units=15,
            challenge="challenge",
            backend="llama.cpp-rpc",
            kernel_micros=1_000,
        )
        self.assertEqual(receipt.output_digest, sha256("output"))
        self.assertTrue(verify_receipt_signature(receipt))


if __name__ == "__main__":
    unittest.main()
