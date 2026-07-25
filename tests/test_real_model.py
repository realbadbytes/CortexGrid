from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cortexgrid.network import CortexNetwork


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_REAL_MODEL_TESTS = os.environ.get("CORTEX_TEST_REAL_MODEL") == "1"


@unittest.skipUnless(
    RUN_REAL_MODEL_TESTS,
    "set CORTEX_TEST_REAL_MODEL=1 to run the 100 MB distributed model test",
)
class RealDistributedModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.network = CortexNetwork(
            executor_path=PROJECT_ROOT / "build" / "cortex_worker",
            data_dir=Path(self.temp_directory.name),
            real_model=True,
        )

    def tearDown(self) -> None:
        self.network.close()
        self.temp_directory.cleanup()

    def test_real_model_runs_on_two_nodes_and_finalizes_block(self) -> None:
        prompt = "What is distributed inference? Answer in one sentence."
        result = self.network.run_inference(prompt)

        self.assertFalse(result["simulation"])
        self.assertEqual(result["runtime"]["mode"], "real-transformer")
        self.assertEqual(
            result["runtime"]["proofLevel"],
            "signed-participation-attestation",
        )
        self.assertEqual(len(result["route"]), 2)
        self.assertEqual(len({stage["nodeId"] for stage in result["route"]}), 2)
        self.assertEqual(len(result["proof"]["receipts"]), 2)
        self.assertGreater(result["performance"]["tokens"]["completion_tokens"], 0)
        self.assertTrue(result["output"]["text"])
        self.assertTrue(self.network.chain.validate()["valid"])
        reference = self.network.model_runtime.generate(
            prompt=prompt,
            route=["atlas"],
        )
        self.assertEqual(result["output"]["text"], reference["text"])

    def test_real_route_reconfigures_to_replica(self) -> None:
        self.network.set_node_status("atlas", "offline")
        result = self.network.run_inference(
            "Name one benefit of model replication."
        )

        self.assertNotIn("atlas", {stage["nodeId"] for stage in result["route"]})
        self.assertEqual(result["route"][0]["nodeId"], "kite")
        self.assertTrue(self.network.chain.validate()["valid"])


if __name__ == "__main__":
    unittest.main()
