from __future__ import annotations

import unittest

from cortexgrid.economics import (
    BOOTSTRAP_HALVING_INFERENCES,
    BOOTSTRAP_INITIAL_MATCH_BPS,
    ECONOMICS_VERSION,
    LEGACY_BLOCK_ECONOMICS_VERSION,
    LEGACY_ECONOMICS_VERSION,
    LEGACY_FIXED_WORK_ECONOMICS_VERSION,
    REWARD_RATE_MICRO_PER_WORK_UNIT,
    SERVICE_PROTOCOL_BPS,
    SERVICE_PROVIDER_BPS,
    MAX_SPONSORED_PRICE_MICRO_PER_WORK_UNIT,
    build_economics_record,
)


class EconomicsTests(unittest.TestCase):
    @staticmethod
    def record(
        *,
        inference_index: int = 0,
        output_tokens: int = 10,
        ready_capacity_units: int = 60,
        epoch_fees_before_micro: int = 0,
        provider_rates_micro: dict[str, int] | None = None,
        version: str = ECONOMICS_VERSION,
    ) -> dict:
        return build_economics_record(
            inference_index=inference_index,
            model_compute_units=30,
            input_tokens=10,
            output_tokens=output_tokens,
            kernel_micros=2_000,
            provider_rates_micro=provider_rates_micro
            or {"ctx_a": 10, "ctx_b": 20},
            provider_compute_units={"ctx_a": 10, "ctx_b": 20},
            ready_capacity_units=ready_capacity_units,
            target_capacity_units=60,
            epoch_fees_before_micro=epoch_fees_before_micro,
            epoch_issuance_before_micro=0,
            version=version,
        )

    def test_service_fee_pays_provider_quotes_and_protocol(self) -> None:
        record = self.record()
        pricing = record["pricing"]
        emission = record["emission"]

        self.assertEqual(record["version"], ECONOMICS_VERSION)
        self.assertEqual(
            pricing["providerBps"], SERVICE_PROVIDER_BPS
        )
        self.assertEqual(
            pricing["protocolBps"], SERVICE_PROTOCOL_BPS
        )
        expected_provider_pool = (
            record["metering"]["providerWorkUnits"]["ctx_a"] * 10
            + record["metering"]["providerWorkUnits"]["ctx_b"] * 20
        )
        self.assertEqual(
            pricing["providerPoolMicro"], expected_provider_pool
        )
        self.assertEqual(
            pricing["serviceFeeMicro"],
            pricing["providerPoolMicro"] + pricing["protocolMicro"],
        )
        self.assertEqual(emission["matchBps"], BOOTSTRAP_INITIAL_MATCH_BPS)
        self.assertEqual(
            emission["requestedIssuanceMicro"],
            expected_provider_pool * BOOTSTRAP_INITIAL_MATCH_BPS // 10_000,
        )

    def test_more_verified_processing_earns_more_service_and_subsidy(self) -> None:
        short = self.record(output_tokens=10)
        long = self.record(output_tokens=100)

        self.assertGreater(
            long["metering"]["meteredWorkUnits"],
            short["metering"]["meteredWorkUnits"],
        )
        self.assertGreater(
            long["pricing"]["providerPoolMicro"],
            short["pricing"]["providerPoolMicro"],
        )
        self.assertGreater(
            long["emission"]["requestedIssuanceMicro"],
            short["emission"]["requestedIssuanceMicro"],
        )

    def test_bootstrap_match_declines_but_service_payment_continues(self) -> None:
        early = self.record(
            inference_index=0,
            ready_capacity_units=1,
            epoch_fees_before_micro=0,
        )
        later = self.record(
            inference_index=BOOTSTRAP_HALVING_INFERENCES,
            ready_capacity_units=60,
            epoch_fees_before_micro=1_000_000_000,
        )

        self.assertEqual(early["pricing"], later["pricing"])
        self.assertEqual(
            later["emission"]["matchBps"],
            early["emission"]["matchBps"] // 2,
        )
        self.assertEqual(
            later["emission"]["requestedIssuanceMicro"],
            early["emission"]["requestedIssuanceMicro"] // 2,
        )

    def test_provider_rates_change_paid_service_cost(self) -> None:
        lower = self.record(
            provider_rates_micro={"ctx_a": 5, "ctx_b": 5}
        )
        higher = self.record(
            provider_rates_micro={"ctx_a": 20, "ctx_b": 20}
        )

        self.assertGreater(
            higher["pricing"]["serviceFeeMicro"],
            lower["pricing"]["serviceFeeMicro"],
        )

    def test_sponsor_rejects_unbounded_provider_price(self) -> None:
        with self.assertRaisesRegex(ValueError, "sponsored inference ceiling"):
            self.record(
                provider_rates_micro={
                    "ctx_a": MAX_SPONSORED_PRICE_MICRO_PER_WORK_UNIT + 1,
                    "ctx_b": 20,
                }
            )

    def test_v2_v3_and_v4_policies_remain_replayable(self) -> None:
        v2 = self.record(
            ready_capacity_units=30,
            epoch_fees_before_micro=100_000_000,
            version=LEGACY_ECONOMICS_VERSION,
        )
        v3 = self.record(version=LEGACY_BLOCK_ECONOMICS_VERSION)
        v4 = self.record(version=LEGACY_FIXED_WORK_ECONOMICS_VERSION)

        self.assertEqual(v2["pricing"]["burnBps"], 500)
        self.assertEqual(v2["emission"]["capacityUnlockBps"], 5_000)
        self.assertEqual(v3["pricing"]["burnBps"], 0)
        self.assertIn("blockRewardMicro", v3["emission"])
        self.assertEqual(
            v4["reward"]["requestedIssuanceMicro"],
            v4["metering"]["meteredWorkUnits"]
            * REWARD_RATE_MICRO_PER_WORK_UNIT,
        )


if __name__ == "__main__":
    unittest.main()
