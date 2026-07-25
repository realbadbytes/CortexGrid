from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

from cortexgrid.join_protocol import request_json
from cortexgrid.network import CortexNetwork
from cortexgrid.node_daemon import ProviderNode
from cortexgrid.provider_cli import ProviderOperatorApi
from cortexgrid.server import CortexHttpServer
from cortexgrid.util import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PATH = Path(
    os.environ.get(
        "CORTEX_EXECUTOR_PATH",
        PROJECT_ROOT / "build" / "cortex_worker",
    )
)


class RemoteJoinTests(unittest.TestCase):
    def test_remote_provider_joins_executes_and_receives_ctx(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = CortexNetwork(
                executor_path=EXECUTOR_PATH,
                data_dir=root / "coordinator",
                real_model=False,
            )
            try:
                coordinator = CortexHttpServer(("127.0.0.1", 0), network)
            except PermissionError:
                network.close()
                self.skipTest("local socket binding is disabled in this sandbox")
            coordinator_thread = threading.Thread(
                target=coordinator.serve_forever, daemon=True
            )
            coordinator_thread.start()
            provider = ProviderNode(
                coordinator_url=f"http://127.0.0.1:{coordinator.server_port}",
                node_id="provider-one",
                name="Provider One",
                operator="Test Operator",
                region="local-test",
                memory_gib=8,
                network_latency_ms=1,
                executor_path=EXECUTOR_PATH,
                rpc_binary=None,
                data_dir=root / "provider",
                listen_host="127.0.0.1",
                advertise_host="127.0.0.1",
                api_port=0,
                rpc_port=1,
                heartbeat_seconds=60,
                price_micro_per_work_unit=7,
            )
            try:
                joined = provider.start()
                self.assertEqual(joined["status"], "ready")
                self.assertEqual(joined["assignment"]["shard"]["index"], 0)
                self.assertEqual(
                    joined["assignment"]["economics"][
                        "priceMicroPerWorkUnit"
                    ],
                    7,
                )
                self.assertEqual(
                    joined["rewardAddress"], provider.reward_address
                )
                self.assertEqual(
                    network.node_snapshot("provider-one")["status"], "online"
                )
                with self.assertRaises(ValueError):
                    network.set_node_status("provider-one", "online")

                shard = network.nodes["provider-one"].shards[0]
                remote_worker = network.nodes["provider-one"].worker
                remote_worker.begin_job(
                    request_id="inf_visible_job",
                    model_id=network.model["id"],
                    model_version=network.model["version"],
                    shard_index=shard["index"],
                    input_digest=sha256("visible-input"),
                    weights_digest=shard["weightsDigest"],
                    compute_units=shard["computeUnits"],
                    challenge="1" * 32,
                )
                self.assertEqual(
                    provider.jobs_snapshot()["active"][0]["requestId"],
                    "inf_visible_job",
                )
                remote_worker.cancel_job(
                    request_id="inf_visible_job",
                    model_id=network.model["id"],
                    model_version=network.model["version"],
                    shard_index=shard["index"],
                    input_digest=sha256("visible-input"),
                    weights_digest=shard["weightsDigest"],
                    compute_units=shard["computeUnits"],
                    challenge="2" * 32,
                )

                result = network.run_inference(
                    "Use a dynamically registered remote provider."
                )

                self.assertEqual(result["route"][0]["nodeId"], "provider-one")
                self.assertTrue(network.chain.validate()["valid"])
                balance = network.node_snapshot("provider-one")["balance"]
                self.assertGreater(balance, 0)
                self.assertEqual(
                    network.node_snapshot("provider-one")[
                        "priceMicroPerWorkUnit"
                    ],
                    7,
                )
                heartbeat = provider.heartbeat()
                self.assertEqual(heartbeat["balance"], balance)
                self.assertEqual(heartbeat["completedJobs"], 1)
                coordinator_view = request_json(
                    f"http://127.0.0.1:{coordinator.server_port}"
                    "/api/nodes/provider-one"
                )
                economics_view = request_json(
                    f"http://127.0.0.1:{coordinator.server_port}"
                    "/api/economics"
                )
                self.assertEqual(
                    economics_view["version"], "ctx-economics-v2"
                )
                provider_view = request_json(
                    f"{provider.attestation_endpoint}/v1/balance"
                )
                self.assertEqual(coordinator_view["balance"], balance)
                self.assertEqual(provider_view["balance"], balance)
                finalized = next(
                    job
                    for job in provider_view["recentJobs"]
                    if job["requestId"] == result["id"]
                )
                self.assertEqual(finalized["status"], "finalized")
                self.assertGreater(finalized["reward"], 0)
                self.assertEqual(finalized["blockHeight"], 1)

                operator = ProviderOperatorApi(
                    provider.attestation_endpoint,
                    provider.operator_token,
                )
                self.assertEqual(
                    operator.request("/v1/provider")["nodeId"],
                    "provider-one",
                )
                self.assertTrue(operator.request("/v1/jobs")["recent"])
                left = operator.request("/v1/operator/leave", post=True)
                self.assertEqual(left["status"], "left")
                self.assertEqual(
                    network.node_snapshot("provider-one")["status"], "left"
                )
                rejoined = operator.request("/v1/operator/join", post=True)
                self.assertEqual(rejoined["status"], "ready")
                self.assertEqual(
                    network.node_snapshot("provider-one")["status"], "online"
                )

                rejoined = provider.join_network()
                self.assertEqual(rejoined["status"], "ready")
                self.assertEqual(rejoined["balance"], balance)
                self.assertEqual(
                    network.node_snapshot("provider-one")["completedJobs"], 1
                )

                network.nodes["provider-one"].last_heartbeat_ms = 0
                self.assertEqual(
                    network.node_snapshot("provider-one")["status"], "offline"
                )
                self.assertEqual(provider.heartbeat()["status"], "online")
            finally:
                provider.close()
                coordinator.shutdown()
                coordinator.server_close()
                coordinator_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
