from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

from cortexgrid.blockchain import (
    GENESIS_SUPPLY_MICRO,
    MAX_SUPPLY_MICRO,
    TOKEN_UNIT,
    inference_transactions,
    verify_proof_bundle,
)
from cortexgrid.economics import (
    ECONOMICS_VERSION,
)
from cortexgrid.network import CortexNetwork
from cortexgrid.wallet import Wallet


EXECUTOR_PATH = Path(
    os.environ.get(
        "CORTEX_EXECUTOR_PATH",
        Path(__file__).resolve().parents[1] / "build" / "cortex_worker",
    )
)


class NetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.network = CortexNetwork(
            executor_path=EXECUTOR_PATH,
            data_dir=Path(self.temp_directory.name),
            real_model=False,
        )

    def tearDown(self) -> None:
        self.network.close()
        self.temp_directory.cleanup()

    def test_inference_finalizes_useful_work_block(self) -> None:
        result = self.network.run_inference(
            "How can a model be larger than one machine?"
        )

        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["block"]["height"], 1)
        self.assertEqual(len(result["proof"]["receipts"]), 12)
        self.assertTrue(verify_proof_bundle(result["proof"])["valid"])
        self.assertTrue(self.network.chain.validate()["valid"])
        accounting = result["accounting"]
        self.assertEqual(accounting["economicsVersion"], ECONOMICS_VERSION)
        self.assertEqual(accounting["endUserFee"], 0)
        self.assertFalse(accounting["endUserWalletRequired"])
        economics = self.network.chain.blocks[-1]["payload"]["economics"]
        expected_issuance_micro = economics["emission"][
            "requestedIssuanceMicro"
        ]
        self.assertEqual(
            accounting["issuance"],
            expected_issuance_micro / TOKEN_UNIT,
        )
        self.assertAlmostEqual(
            accounting["providerRewards"],
            accounting["providerServicePayments"] + accounting["issuance"],
            places=6,
        )
        transactions = self.network.chain.blocks[-1]["transactions"]
        self.assertEqual(transactions[0]["type"], "inference_fee")
        self.assertEqual(transactions[1]["type"], "protocol_fee")
        self.assertTrue(
            not any(
                transaction["type"] == "burn"
                for transaction in transactions
            )
        )
        service_fee_micro = transactions[0]["amountMicro"]
        protocol_fee_micro = transactions[1]["amountMicro"]
        provider_fee_micro = sum(
            transaction["feeAmountMicro"]
            for transaction in transactions
            if transaction["type"] == "provider_reward"
        )
        self.assertEqual(
            service_fee_micro,
            protocol_fee_micro + provider_fee_micro,
        )
        self.assertEqual(
            accounting["serviceSponsorFee"],
            service_fee_micro / TOKEN_UNIT,
        )
        issuance_micro = sum(
            transaction["issuanceAmountMicro"]
            for transaction in transactions
            if transaction["type"] == "provider_reward"
        )
        self.assertEqual(
            self.network.chain.total_supply_micro,
            GENESIS_SUPPLY_MICRO + issuance_micro,
        )
        self.assertEqual(issuance_micro, expected_issuance_micro)
        self.assertEqual(
            self.network.chain.snapshot()["token"]["genesisSupply"],
            GENESIS_SUPPLY_MICRO / TOKEN_UNIT,
        )
        self.assertEqual(
            self.network.chain.snapshot()["token"]["totalIssued"],
            expected_issuance_micro / TOKEN_UNIT,
        )
        self.assertEqual(
            economics["metering"]["meteredWorkUnits"],
            sum(economics["metering"]["providerWorkUnits"].values()),
        )
        for transaction in transactions:
            if transaction["type"] == "provider_reward":
                self.assertEqual(
                    transaction["feeAmountMicro"],
                    transaction["quotedAmountMicro"],
                )

    def test_replica_routing_survives_one_offline_node(self) -> None:
        self.network.set_node_status("atlas", "offline")
        result = self.network.run_inference("Test shard replica failover.")

        self.assertEqual(result["status"], "finalized")
        self.assertNotIn("atlas", {stage["nodeId"] for stage in result["route"]})
        self.assertTrue(self.network.chain.validate()["valid"])

    def test_signed_token_transfer_does_not_mint(self) -> None:
        recipient = Wallet.generate()
        starting_supply = self.network.chain.total_supply_micro
        amount = 2 * TOKEN_UNIT
        transaction = self.network.demo_wallet.sign_transfer(
            recipient.address,
            amount,
            self.network.chain.nonce_of(self.network.demo_wallet.address),
        )

        block = self.network.chain.commit_transfer(transaction)

        self.assertEqual(block["kind"], "transfer")
        self.assertEqual(self.network.chain.balance_of(recipient.address), amount)
        self.assertEqual(self.network.chain.total_supply_micro, starting_supply)
        self.assertTrue(self.network.chain.validate()["valid"])

    def test_useful_work_reward_is_clipped_at_supply_cap(self) -> None:
        result = self.network.run_inference(
            "Create a valid proof for maximum supply testing."
        )
        economics = self.network.chain.blocks[-1]["payload"]["economics"]

        transactions = inference_transactions(
            proof=result["proof"],
            payer=self.network.demo_wallet.address,
            inference_id="inf_cap_test",
            current_supply_micro=MAX_SUPPLY_MICRO - 1,
            economics=economics,
        )
        issued = sum(
            transaction["issuanceAmountMicro"]
            for transaction in transactions
            if transaction["type"] == "provider_reward"
        )
        paid_service = sum(
            transaction["feeAmountMicro"]
            for transaction in transactions
            if transaction["type"] == "provider_reward"
        )

        self.assertEqual(issued, 1)
        self.assertGreater(paid_service, 0)

    def test_consensus_rejects_forged_reward_amount(self) -> None:
        self.network.run_inference("Create a block for tamper testing.")
        original = copy.deepcopy(self.network.chain.blocks[-1])
        forged = self.network.chain.blocks[-1]
        reward = next(
            transaction
            for transaction in forged["transactions"]
            if transaction["type"] == "provider_reward"
        )
        reward["issuanceAmountMicro"] += 1
        forged["hash"] = self.network.chain._block_hash(forged)

        validation = self.network.chain.validate()

        self.assertFalse(validation["valid"])
        self.assertIn("settlement", validation["reason"])
        self.network.chain.blocks[-1] = original

    def test_consensus_rejects_forged_service_payment(self) -> None:
        self.network.run_inference("Create a priced block for tamper testing.")
        original = copy.deepcopy(self.network.chain.blocks[-1])
        forged = self.network.chain.blocks[-1]
        forged["transactions"][0]["amountMicro"] += 1
        forged["hash"] = self.network.chain._block_hash(forged)

        validation = self.network.chain.validate()

        self.assertFalse(validation["valid"])
        self.assertIn("settlement", validation["reason"])
        self.network.chain.blocks[-1] = original


if __name__ == "__main__":
    unittest.main()
