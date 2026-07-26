from __future__ import annotations

import unittest

from cortexgrid.economics import (
    ECONOMICS_VERSION,
    LEGACY_BLOCK_ECONOMICS_VERSION,
    LEGACY_ECONOMICS_VERSION,
    REWARD_RATE_MICRO_PER_WORK_UNIT,
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

    def test_reward_is_fixed_rate_times_verified_work(self) -> None:
        record = self.record()
        total_work = record["metering"]["meteredWorkUnits"]

        self.assertEqual(record["version"], ECONOMICS_VERSION)
        self.assertEqual(
            record["reward"]["rateMicroPerWorkUnit"],
            REWARD_RATE_MICRO_PER_WORK_UNIT,
        )
        self.assertEqual(
            record["reward"]["requestedIssuanceMicro"],
            total_work * REWARD_RATE_MICRO_PER_WORK_UNIT,
        )
        self.assertNotIn("pricing", record)
        self.assertNotIn("emission", record)

    def test_more_verified_processing_earns_more_ctx(self) -> None:
        short = self.record(output_tokens=10)
        long = self.record(output_tokens=100)

        self.assertGreater(
            long["metering"]["meteredWorkUnits"],
            short["metering"]["meteredWorkUnits"],
        )
        self.assertGreater(
            long["reward"]["requestedIssuanceMicro"],
            short["reward"]["requestedIssuanceMicro"],
        )

    def test_capacity_demand_price_and_block_count_do_not_change_rate(
        self,
    ) -> None:
        first = self.record(
            inference_index=0,
            ready_capacity_units=1,
            epoch_fees_before_micro=0,
            provider_rates_micro={"ctx_a": 1, "ctx_b": 1},
        )
        later = self.record(
            inference_index=9_999_999,
            ready_capacity_units=60,
            epoch_fees_before_micro=1_000_000_000,
            provider_rates_micro={"ctx_a": 9_999, "ctx_b": 9_999},
        )

        self.assertEqual(first, later)

    def test_v2_and_v3_policies_remain_replayable(self) -> None:
        v2 = self.record(
            ready_capacity_units=30,
            epoch_fees_before_micro=100_000_000,
            version=LEGACY_ECONOMICS_VERSION,
        )
        v3 = self.record(version=LEGACY_BLOCK_ECONOMICS_VERSION)

        self.assertEqual(v2["pricing"]["burnBps"], 500)
        self.assertEqual(v2["emission"]["capacityUnlockBps"], 5_000)
        self.assertEqual(v3["pricing"]["burnBps"], 0)
        self.assertIn("blockRewardMicro", v3["emission"])


if __name__ == "__main__":
    unittest.main()
