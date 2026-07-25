from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

from cortexgrid.network import CortexNetwork
from cortexgrid.node_daemon import ProviderNode
from cortexgrid.server import CortexHttpServer


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
        economics = result["block"]["payload"]["economics"]
        self.assertEqual(
            economics["metering"]["inputTokens"],
            result["performance"]["tokens"]["prompt_tokens"],
        )
        self.assertEqual(
            economics["metering"]["outputTokens"],
            result["performance"]["tokens"]["completion_tokens"],
        )
        self.assertGreater(result["accounting"]["sponsorFee"], 0)
        self.assertGreater(result["accounting"]["providerFeeRewards"], 0)
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

    def test_remote_provider_joins_real_model_route_and_gets_paid(self) -> None:
        coordinator = CortexHttpServer(("127.0.0.1", 0), self.network)
        coordinator_thread = threading.Thread(
            target=coordinator.serve_forever, daemon=True
        )
        coordinator_thread.start()
        provider = ProviderNode(
            coordinator_url=f"http://127.0.0.1:{coordinator.server_port}",
            node_id="remote-gpu-one",
            name="Remote GPU One",
            operator="Independent",
            region="test-remote",
            memory_gib=8,
            network_latency_ms=1,
            executor_path=PROJECT_ROOT / "build" / "cortex_worker",
            rpc_binary=PROJECT_ROOT
            / "build"
            / "llama"
            / "bin"
            / "ggml-rpc-server",
            data_dir=Path(self.temp_directory.name) / "remote-provider",
            listen_host="127.0.0.1",
            advertise_host="127.0.0.1",
            api_port=0,
            rpc_port=0,
            heartbeat_seconds=60,
        )
        try:
            joined = provider.start()
            self.assertEqual(joined["assignment"]["shard"]["index"], 0)

            result = self.network.run_inference(
                "Explain why model sharding is useful in one sentence."
            )

            self.assertEqual(result["route"][0]["nodeId"], "remote-gpu-one")
            self.assertFalse(result["simulation"])
            self.assertTrue(self.network.chain.validate()["valid"])
            self.assertGreater(
                self.network.node_snapshot("remote-gpu-one")["balance"], 0
            )
            participant = next(
                item
                for item in self.network.model_runtime.snapshot()["participants"]
                if item["nodeId"] == "remote-gpu-one"
            )
            self.assertTrue(participant["remote"])
            self.assertTrue(participant["resident"])
        finally:
            provider.close()
            coordinator.shutdown()
            coordinator.server_close()
            coordinator_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
