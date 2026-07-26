from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from cortexgrid.join_protocol import request_json
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


class ParticipationTests(unittest.TestCase):
    def test_public_inference_requires_online_interactive_provider(self) -> None:
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
            thread = threading.Thread(
                target=coordinator.serve_forever, daemon=True
            )
            thread.start()
            coordinator_url = (
                f"http://127.0.0.1:{coordinator.server_port}"
            )
            provider = ProviderNode(
                coordinator_url=coordinator_url,
                node_id="term-participant",
                name="Interactive Participant",
                operator="Test User",
                region="local-test",
                memory_gib=8,
                network_latency_ms=1,
                executor_path=EXECUTOR_PATH,
                rpc_binary=None,
                data_dir=root / "participant",
                listen_host="127.0.0.1",
                advertise_host="127.0.0.1",
                api_port=0,
                rpc_port=1,
                heartbeat_seconds=60,
                role="interactive-user",
            )
            try:
                provider.start()
                snapshot = request_json(f"{coordinator_url}/api/network")
                self.assertEqual(
                    snapshot["metrics"]["interactiveUsersOnline"], 1
                )
                self.assertEqual(
                    snapshot["metrics"]["dedicatedWorkersOnline"], 0
                )

                bridge_origin = coordinator_url
                bridge_url = (
                    f"{provider.attestation_endpoint}/v1/browser/status"
                )
                preflight = urllib.request.Request(
                    bridge_url,
                    method="OPTIONS",
                    headers={
                        "Origin": bridge_origin,
                        "Access-Control-Request-Method": "GET",
                        "Access-Control-Request-Headers": "authorization",
                        "Access-Control-Request-Private-Network": "true",
                    },
                )
                with urllib.request.urlopen(preflight, timeout=5) as response:
                    self.assertEqual(response.status, 204)
                    self.assertEqual(
                        response.headers["Access-Control-Allow-Origin"],
                        bridge_origin,
                    )
                    self.assertEqual(
                        response.headers[
                            "Access-Control-Allow-Private-Network"
                        ],
                        "true",
                    )

                unauthorized = urllib.request.Request(
                    bridge_url,
                    headers={"Origin": bridge_origin},
                )
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(unauthorized, timeout=5)
                self.assertEqual(denied.exception.code, 401)

                status_request = urllib.request.Request(
                    bridge_url,
                    headers={
                        "Authorization": f"Bearer {provider.browser_token}",
                        "Origin": bridge_origin,
                    },
                )
                with urllib.request.urlopen(
                    status_request, timeout=5
                ) as response:
                    browser_status = json.loads(response.read())
                    self.assertEqual(
                        response.headers["Access-Control-Allow-Origin"],
                        bridge_origin,
                    )
                self.assertTrue(browser_status["connected"])
                self.assertEqual(
                    browser_status["provider"]["nodeId"], provider.node_id
                )

                browser_prompt = "Run this from CortexGrid Chat."
                browser_request = urllib.request.Request(
                    (
                        f"{provider.attestation_endpoint}"
                        "/v1/browser/inference"
                    ),
                    data=json.dumps({"prompt": browser_prompt}).encode(),
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {provider.browser_token}",
                        "Content-Type": "application/json",
                        "Origin": bridge_origin,
                    },
                )
                with urllib.request.urlopen(
                    browser_request, timeout=30
                ) as response:
                    browser_result = json.loads(response.read())
                self.assertEqual(browser_result["status"], "finalized")
                self.assertEqual(
                    browser_result["access"]["participantNodeId"],
                    provider.node_id,
                )
                self.assertGreater(
                    browser_result["access"]["participantReward"], 0
                )

                with self.assertRaisesRegex(
                    RuntimeError, "compute participant"
                ):
                    request_json(
                        f"{coordinator_url}/api/inference",
                        payload={"prompt": "No participation proof."},
                    )

                prompt = "Use the pooled model while contributing compute."
                authorization = provider.authorize_inference(prompt)
                with self.assertRaisesRegex(
                    RuntimeError, "does not match"
                ):
                    request_json(
                        f"{coordinator_url}/api/inference",
                        payload={
                            "prompt": "A different prompt.",
                            "participation": authorization,
                        },
                    )

                result = request_json(
                    f"{coordinator_url}/api/inference",
                    payload={
                        "prompt": prompt,
                        "participation": authorization,
                    },
                    timeout=30,
                )
                self.assertEqual(result["status"], "finalized")
                self.assertEqual(
                    result["access"]["participantNodeId"],
                    provider.node_id,
                )
                self.assertTrue(
                    result["access"]["computeContributionRequired"]
                )
                self.assertEqual(
                    result["access"]["mode"], "compute-participant"
                )
                self.assertTrue(result["access"]["participantSelected"])
                self.assertGreater(
                    result["access"]["participantReward"], 0
                )
                self.assertIn(
                    provider.node_id,
                    {stage["nodeId"] for stage in result["route"]},
                )
                self.assertGreater(
                    network.node_snapshot(provider.node_id)["balance"], 0
                )

                with self.assertRaisesRegex(
                    RuntimeError, "already been used"
                ):
                    request_json(
                        f"{coordinator_url}/api/inference",
                        payload={
                            "prompt": prompt,
                            "participation": authorization,
                        },
                    )

                offline_prompt = "This must fail after CortexGrid Chat leaves."
                offline_authorization = provider.authorize_inference(
                    offline_prompt
                )
                provider.leave_network()
                with self.assertRaisesRegex(RuntimeError, "not online"):
                    request_json(
                        f"{coordinator_url}/api/inference",
                        payload={
                            "prompt": offline_prompt,
                            "participation": offline_authorization,
                        },
                    )
            finally:
                provider.close()
                coordinator.shutdown()
                coordinator.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
