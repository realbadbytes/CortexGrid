from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

from cortexgrid.model_profiles import (
    FIVE_TIB_BYTES,
    capacity_plan,
    capacity_requirements,
    five_tb_model_manifest,
)
from cortexgrid.network import CortexNetwork
from cortexgrid.node_daemon import ProviderNode
from cortexgrid.server import CortexHttpServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PATH = Path(
    os.environ.get(
        "CORTEX_EXECUTOR_PATH",
        PROJECT_ROOT / "build" / "cortex_worker",
    )
)


class FiveTiBModelProfileTests(unittest.TestCase):
    def test_capacity_profile_is_exactly_five_tib(self) -> None:
        manifest = five_tb_model_manifest()
        requirements = capacity_requirements(manifest)

        self.assertEqual(manifest["modelSizeBytes"], FIVE_TIB_BYTES)
        self.assertEqual(manifest["shardCount"], 160)
        self.assertEqual(len(manifest["shards"]), 160)
        self.assertTrue(
            all(shard["sizeGiB"] == 32 for shard in manifest["shards"])
        )
        self.assertTrue(
            all(
                shard["minimumMemoryGiB"] == 40
                for shard in manifest["shards"]
            )
        )
        self.assertEqual(requirements["minimumProvidersNoReplication"], 160)
        self.assertEqual(
            requirements["minimumProvidersAtTargetReplication"], 320
        )
        self.assertEqual(requirements["weightMemoryGiBNoReplication"], 5120)
        self.assertEqual(
            requirements["minimumNetworkRamGiBNoReplication"], 6400
        )

    def test_capacity_plan_requires_compatible_providers(self) -> None:
        manifest = five_tb_model_manifest()

        undersized = capacity_plan(
            manifest, provider_count=320, provider_memory_gib=32
        )
        one_copy = capacity_plan(
            manifest, provider_count=160, provider_memory_gib=40
        )
        redundant = capacity_plan(
            manifest, provider_count=320, provider_memory_gib=40
        )

        self.assertFalse(undersized["readyForOneReplica"])
        self.assertEqual(undersized["stagePlacementsPossible"], 0)
        self.assertTrue(one_copy["readyForOneReplica"])
        self.assertFalse(one_copy["readyForTargetReplication"])
        self.assertTrue(redundant["readyForTargetReplication"])

    def test_network_refuses_inference_until_all_stages_are_hosted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            network = CortexNetwork(
                executor_path=EXECUTOR_PATH,
                data_dir=Path(directory),
                model_profile="5tb",
            )
            try:
                model = network.public_model()
                self.assertEqual(model["modelSizeBytes"], FIVE_TIB_BYTES)
                self.assertEqual(model["coverage"]["readyStageCount"], 0)
                self.assertEqual(model["coverage"]["missingStageCount"], 160)
                self.assertEqual(model["status"], "awaiting-capacity")
                self.assertEqual(network.snapshot()["metrics"]["modelsReady"], 0)
                with self.assertRaisesRegex(
                    RuntimeError, "awaiting 160 unhosted stage"
                ):
                    network.run_inference("This route is not ready.")
            finally:
                network.close()

    def test_compatible_provider_receives_one_exact_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = CortexNetwork(
                executor_path=EXECUTOR_PATH,
                data_dir=root / "controller",
                model_profile="5tb",
            )
            coordinator = CortexHttpServer(("127.0.0.1", 0), network)
            coordinator_thread = threading.Thread(
                target=coordinator.serve_forever, daemon=True
            )
            coordinator_thread.start()
            provider = ProviderNode(
                coordinator_url=(
                    f"http://127.0.0.1:{coordinator.server_port}"
                ),
                node_id="capacity-provider",
                name="Capacity Provider",
                operator="Test",
                region="local",
                memory_gib=40,
                network_latency_ms=1,
                executor_path=EXECUTOR_PATH,
                rpc_binary=None,
                data_dir=root / "provider",
                listen_host="127.0.0.1",
                advertise_host="127.0.0.1",
                api_port=0,
                rpc_port=1,
                heartbeat_seconds=60,
            )
            try:
                joined = provider.start()
                shard = joined["assignment"]["shard"]

                self.assertEqual(shard["index"], 0)
                self.assertEqual(shard["sizeBytes"], 32 * 1024**3)
                self.assertEqual(shard["minimumMemoryGiB"], 40)
                self.assertEqual(
                    shard["artifact"]["sha256"], shard["weightsDigest"]
                )
                self.assertFalse(shard["artifact"]["available"])
                self.assertEqual(
                    network.public_model()["coverage"]["readyStageCount"], 1
                )
                node = network.node_snapshot("capacity-provider")
                self.assertEqual(node["allocatedMemoryGiB"], 32)
                self.assertEqual(node["reservedMemoryGiB"], 40)
                self.assertEqual(node["freeMemoryGiB"], 0)
            finally:
                provider.close()
                coordinator.shutdown()
                coordinator.server_close()
                coordinator_thread.join(timeout=2)

    def test_replacement_assignment_prioritizes_an_offline_stage(self) -> None:
        class Placement:
            def __init__(self, stage: int, status: str = "online"):
                self.stage = stage
                self.status = status

            def hosts(self, model_id: str, shard_index: int) -> bool:
                return (
                    model_id == "cortex-5tb-theoretical"
                    and shard_index == self.stage
                )

        with tempfile.TemporaryDirectory() as directory:
            network = CortexNetwork(
                executor_path=EXECUTOR_PATH,
                data_dir=Path(directory),
                model_profile="5tb",
            )
            fake_ids = []
            try:
                for stage in range(160):
                    fake_id = f"fake-{stage}"
                    fake_ids.append(fake_id)
                    network.nodes[fake_id] = Placement(
                        stage, "offline" if stage == 73 else "online"
                    )

                replacement = network._choose_remote_shard(40)

                self.assertEqual(replacement["index"], 73)
            finally:
                for fake_id in fake_ids:
                    network.nodes.pop(fake_id, None)
                network.close()


if __name__ == "__main__":
    unittest.main()
