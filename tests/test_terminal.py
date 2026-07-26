from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from cortexgrid.terminal import CortexSession


class FakeApi:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.participations: list[dict] = []

    def network(self) -> dict:
        return {
            "name": "CortexGrid",
            "environment": "testnet",
            "status": "healthy",
            "metrics": {
                "nodesOnline": 2,
                "nodesTotal": 2,
                "chainHeight": 4,
            },
            "model": {
                "name": "Test Model",
                "precision": "Q4",
                "shardCount": 2,
            },
        }

    def infer(
        self,
        prompt: str,
        model_id: str | None,
        participation: dict,
    ) -> dict:
        self.prompts.append(prompt)
        self.participations.append(participation)
        return {
            "id": f"inf_{len(self.prompts)}",
            "output": {"text": f"answer {len(self.prompts)}"},
            "access": {
                "mode": "compute-participant",
                "endUserWalletRequired": False,
                "endUserFee": 0,
                "symbol": "CTX",
                "participantSelected": True,
                "participantReward": 0.5,
            },
            "route": [
                {"stage": 0, "nodeId": "atlas", "nodeName": "Atlas"},
                {"stage": 1, "nodeId": "ember", "nodeName": "Ember"},
            ],
            "proof": {"root": "a" * 64},
            "block": {"height": 5, "hash": "b" * 64},
            "accounting": {
                "providerRewards": 1.0,
                "serviceSponsorFee": 0.4,
                "verifiedWorkUnits": 10_000,
                "symbol": "CTX",
            },
        }


class CortexChatTests(unittest.TestCase):
    @staticmethod
    def signer(prompt: str, model_id: str | None) -> dict:
        return {
            "request": {"prompt": prompt, "modelId": model_id},
            "signature": "test-signature",
        }

    def test_conversation_context_and_sponsored_result(self) -> None:
        api = FakeApi()
        session = CortexSession(
            api, color=False, participation_signer=self.signer
        )

        first = session.ask("Hello")
        second = session.ask("What did I say?")

        self.assertEqual(first["access"]["endUserFee"], 0)
        self.assertEqual(api.prompts[0], "Hello")
        self.assertIn("User: Hello", api.prompts[1])
        self.assertIn("Assistant: answer 1", api.prompts[1])
        self.assertTrue(api.prompts[1].endswith("Assistant:"))
        self.assertEqual(
            api.participations[1]["request"]["prompt"], api.prompts[1]
        )
        self.assertEqual(second["output"]["text"], "answer 2")

    def test_terminal_prints_route_block_and_provider_reward(self) -> None:
        api = FakeApi()
        session = CortexSession(
            api, color=False, participation_signer=self.signer
        )
        result = session.ask("Test")
        output = io.StringIO()

        with redirect_stdout(output):
            session.print_result(result)

        rendered = output.getvalue()
        self.assertIn("Atlas[0] → Ember[1]", rendered)
        self.assertIn("block  #5", rendered)
        self.assertIn("1.0 CTX to compute providers", rendered)
        self.assertIn("10000 verified work units", rendered)
        self.assertIn("earned 0.5 CTX on your node", rendered)
        self.assertIn("requester wallet 0 CTX", rendered)
        self.assertIn("gateway sponsored 0.4 CTX", rendered)

    def test_inference_requires_compute_participation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "compute participant"):
            CortexSession(FakeApi(), color=False).ask("Test")

    def test_status_uses_live_network_snapshot(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            CortexSession(FakeApi(), color=False).print_status()
        self.assertIn("Providers: 2/2 online", output.getvalue())


if __name__ == "__main__":
    unittest.main()
