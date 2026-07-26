from __future__ import annotations

import unittest

from cortexgrid.provider_cli import format_jobs, format_status


class ProviderCliTests(unittest.TestCase):
    def test_status_shows_assignment_capacity_and_balance(self) -> None:
        rendered = format_status(
            {
                "name": "Worker One",
                "nodeId": "worker-one",
                "joined": True,
                "operator": "Independent",
                "region": "us-east",
                "rpcOnline": True,
                "rpcEndpoint": "10.0.0.2:50052",
                "assignment": {
                    "model": {"name": "Model", "version": "q4"},
                    "shard": {
                        "index": 1,
                        "layerRange": [15, 29],
                        "weightsDigest": "a" * 64,
                    },
                    "economics": {
                        "version": "ctx-economics-v4",
                        "rewardDistribution": (
                            "fixed rate multiplied by this node's verified work"
                        ),
                    },
                },
                "activeJobs": [{"requestId": "inf_active"}],
                "completedJobs": 7,
                "priceMicroPerWorkUnit": 12,
                "balance": 3.5,
                "rewardAddress": "ctx_test",
                "chainHeight": 9,
                "lastHeartbeatAt": "now",
                "lastError": None,
            }
        )
        self.assertIn("Stage: 1 · Layers: 15–29", rendered)
        self.assertIn("Jobs: 1 active · 7 completed", rendered)
        self.assertNotIn("Bid:", rendered)
        self.assertIn("fixed rate multiplied", rendered)
        self.assertIn("ctx-economics-v4", rendered)
        self.assertIn("CTX: 3.5", rendered)

    def test_jobs_show_active_and_finalized_rewards(self) -> None:
        rendered = format_jobs(
            {
                "active": [
                    {
                        "requestId": "inf_active",
                        "stage": 0,
                        "status": "running",
                        "startedAt": "now",
                    }
                ],
                "recent": [
                    {
                        "requestId": "inf_done",
                        "stage": 0,
                        "status": "finalized",
                        "reward": 0.5,
                        "blockHeight": 4,
                    }
                ],
            }
        )
        self.assertIn("inf_active · stage 0 · running", rendered)
        self.assertIn("inf_done · finalized · stage 0 · 0.5 CTX · block #4", rendered)


if __name__ == "__main__":
    unittest.main()
