from __future__ import annotations

import unittest

from cortexgrid.economics import (
    EMISSION_HALVING_EPOCHS,
    INITIAL_EPOCH_EMISSION_MICRO,
    build_economics_record,
    scheduled_epoch_emission_micro,
)


class EconomicsTests(unittest.TestCase):
    @staticmethod
    def record(
        *,
        output_tokens: int = 10,
        ready_capacity_units: int = 60,
        epoch_fees_before_micro: int = 100_000_000,
        epoch_issuance_before_micro: int = 0,
    ) -> dict:
        return build_economics_record(
            inference_index=0,
            model_compute_units=30,
            input_tokens=10,
            output_tokens=output_tokens,
            kernel_micros=2_000,
            provider_rates_micro={"ctx_a": 10, "ctx_b": 20},
            provider_compute_units={"ctx_a": 10, "ctx_b": 20},
            ready_capacity_units=ready_capacity_units,
            target_capacity_units=60,
            epoch_fees_before_micro=epoch_fees_before_micro,
            epoch_issuance_before_micro=epoch_issuance_before_micro,
        )

    def test_more_output_work_costs_more(self) -> None:
        short = self.record(output_tokens=10)
        long = self.record(output_tokens=100)

        self.assertGreater(
            long["pricing"]["feeMicro"], short["pricing"]["feeMicro"]
        )
        self.assertGreater(
            long["metering"]["meteredWorkUnits"],
            short["metering"]["meteredWorkUnits"],
        )

    def test_capacity_unlocks_but_does_not_receive_idle_reward(self) -> None:
        half = self.record(ready_capacity_units=30)
        full = self.record(ready_capacity_units=60)

        self.assertEqual(half["emission"]["capacityUnlockBps"], 5_000)
        self.assertEqual(full["emission"]["capacityUnlockBps"], 10_000)
        self.assertLess(
            half["emission"]["requestedIssuanceMicro"],
            full["emission"]["requestedIssuanceMicro"],
        )

    def test_epoch_budget_and_halving_are_deterministic(self) -> None:
        self.assertEqual(
            scheduled_epoch_emission_micro(0),
            INITIAL_EPOCH_EMISSION_MICRO,
        )
        self.assertEqual(
            scheduled_epoch_emission_micro(EMISSION_HALVING_EPOCHS),
            INITIAL_EPOCH_EMISSION_MICRO // 2,
        )
        exhausted = self.record(
            epoch_issuance_before_micro=INITIAL_EPOCH_EMISSION_MICRO
        )
        self.assertEqual(exhausted["emission"]["epochRemainingMicro"], 0)


if __name__ == "__main__":
    unittest.main()
