from __future__ import annotations


TOKEN_UNIT = 1_000_000
TOKEN_SYMBOL = "CTX"
MAX_SUPPLY_MICRO = 1_000_000_000 * TOKEN_UNIT
GENESIS_SUPPLY_MICRO = 1_000_000 * TOKEN_UNIT

LEGACY_ECONOMICS_VERSION = "ctx-economics-v2"
LEGACY_BLOCK_ECONOMICS_VERSION = "ctx-economics-v3"
ECONOMICS_VERSION = "ctx-economics-v4"

PREFILL_WEIGHT = 1
DECODE_WEIGHT = 2

# One rule for all new rewards: 100 micro-CTX for each signed, verified work
# unit. One work unit is one model compute unit applied to one weighted token.
REWARD_RATE_MICRO_PER_WORK_UNIT = 100

# Provider prices remain accepted by the join protocol so old clients can
# connect, but v4 reward calculation does not use them.
PRICE_MIN_MICRO_PER_WORK_UNIT = 1
PRICE_MAX_MICRO_PER_WORK_UNIT = 10_000
DEFAULT_PRICE_MICRO_PER_WORK_UNIT = 10

# v3 constants are retained only to replay existing devnet blocks.
V3_MIN_PROVIDER_POOL_MICRO = 50_000
V3_BURN_BPS = 0
V3_PROTOCOL_BPS = 1_000
V3_PROVIDER_BPS = 9_000
V3_WORK_REWARD_RESERVE_MICRO = MAX_SUPPLY_MICRO - GENESIS_SUPPLY_MICRO
V3_WORK_REWARD_HALVING_INFERENCES = 1_000_000
V3_INITIAL_WORK_REWARD_MICRO = V3_WORK_REWARD_RESERVE_MICRO // (
    2 * V3_WORK_REWARD_HALVING_INFERENCES
)

# v2 constants are retained only to replay existing devnet blocks.
INFERENCES_PER_EPOCH = 1_000
EMISSION_HALVING_EPOCHS = 100
INITIAL_EPOCH_EMISSION_MICRO = 100 * TOKEN_UNIT
MAX_SUBSIDY_BPS_OF_PROVIDER_FEES = 2_000
EPOCH_DEMAND_TARGET_MICRO = 100 * TOKEN_UNIT
CAPACITY_TARGET_REPLICAS = 2
V2_MIN_PROVIDER_POOL_MICRO = 50_000
V2_BURN_BPS = 500
V2_PROTOCOL_BPS = 1_000
V2_PROVIDER_BPS = 8_500


def tokens(amount_micro: int) -> float:
    return round(amount_micro / TOKEN_UNIT, 6)


def ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_div requires a non-negative numerator and denominator")
    return (numerator + denominator - 1) // denominator


def scheduled_epoch_emission_micro(epoch: int) -> int:
    """Return the historical v2 epoch ceiling for ledger replay."""
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    halvings = epoch // EMISSION_HALVING_EPOCHS
    return INITIAL_EPOCH_EMISSION_MICRO >> halvings


def scheduled_work_reward_micro(inference_index: int) -> int:
    """Return the historical v3 block reward for ledger replay."""
    if inference_index < 0:
        raise ValueError("inference index cannot be negative")
    halvings = inference_index // V3_WORK_REWARD_HALVING_INFERENCES
    return V3_INITIAL_WORK_REWARD_MICRO >> halvings


def metered_work_units(
    compute_units: int, input_tokens: int, output_tokens: int
) -> int:
    if compute_units <= 0:
        raise ValueError("compute units must be positive")
    if input_tokens <= 0 or output_tokens < 0:
        raise ValueError("token counts are invalid")
    return compute_units * (
        input_tokens * PREFILL_WEIGHT + output_tokens * DECODE_WEIGHT
    )


def validate_price(price_micro_per_work_unit: object) -> int:
    if isinstance(price_micro_per_work_unit, bool):
        raise ValueError("provider price must be an integer")
    try:
        price = int(price_micro_per_work_unit)
    except (TypeError, ValueError) as error:
        raise ValueError("provider price must be an integer") from error
    if not PRICE_MIN_MICRO_PER_WORK_UNIT <= price <= PRICE_MAX_MICRO_PER_WORK_UNIT:
        raise ValueError(
            "provider price is outside the accepted micro-CTX/work-unit range"
        )
    return price


def _legacy_pricing(
    *,
    provider_work: dict[str, int],
    provider_rates_micro: dict[str, int],
    minimum_provider_pool_micro: int,
    burn_bps: int,
    protocol_bps: int,
    provider_bps: int,
) -> dict:
    normalized_rates = {
        address: validate_price(provider_rates_micro[address])
        for address in sorted(provider_work)
    }
    quoted_provider_cost_micro = sum(
        provider_work[address] * normalized_rates[address]
        for address in provider_work
    )
    target_provider_pool_micro = max(
        minimum_provider_pool_micro, quoted_provider_cost_micro
    )
    fee_micro = ceil_div(target_provider_pool_micro * 10_000, provider_bps)
    burn_micro = fee_micro * burn_bps // 10_000
    protocol_micro = fee_micro * protocol_bps // 10_000
    provider_pool_micro = fee_micro - burn_micro - protocol_micro
    return {
        "providerRatesMicroPerWorkUnit": normalized_rates,
        "quotedProviderCostMicro": quoted_provider_cost_micro,
        "minimumProviderPoolMicro": minimum_provider_pool_micro,
        "feeMicro": fee_micro,
        "burnBps": burn_bps,
        "protocolBps": protocol_bps,
        "providerBps": provider_bps,
        "burnMicro": burn_micro,
        "protocolMicro": protocol_micro,
        "providerPoolMicro": provider_pool_micro,
    }


def build_economics_record(
    *,
    inference_index: int,
    model_compute_units: int,
    input_tokens: int,
    output_tokens: int,
    kernel_micros: int,
    provider_rates_micro: dict[str, int],
    provider_compute_units: dict[str, int],
    ready_capacity_units: int,
    target_capacity_units: int,
    epoch_fees_before_micro: int,
    epoch_issuance_before_micro: int,
    version: str = ECONOMICS_VERSION,
) -> dict:
    if inference_index < 0:
        raise ValueError("inference index cannot be negative")
    if kernel_micros < 0:
        raise ValueError("kernel time cannot be negative")
    if set(provider_rates_micro) != set(provider_compute_units):
        raise ValueError("every participating provider needs exactly one identity")
    if sum(provider_compute_units.values()) != model_compute_units:
        raise ValueError("provider work does not equal model work")

    provider_work = {
        address: metered_work_units(
            int(provider_compute_units[address]), input_tokens, output_tokens
        )
        for address in sorted(provider_compute_units)
    }
    total_work = sum(provider_work.values())
    metering = {
        "modelComputeUnits": model_compute_units,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "kernelMicros": kernel_micros,
        "prefillWeight": PREFILL_WEIGHT,
        "decodeWeight": DECODE_WEIGHT,
        "meteredWorkUnits": total_work,
        "providerWorkUnits": provider_work,
    }

    if version == ECONOMICS_VERSION:
        return {
            "version": version,
            "metering": metering,
            "reward": {
                "rateMicroPerWorkUnit": REWARD_RATE_MICRO_PER_WORK_UNIT,
                "requestedIssuanceMicro": (
                    total_work * REWARD_RATE_MICRO_PER_WORK_UNIT
                ),
            },
        }

    if version == LEGACY_BLOCK_ECONOMICS_VERSION:
        pricing = _legacy_pricing(
            provider_work=provider_work,
            provider_rates_micro=provider_rates_micro,
            minimum_provider_pool_micro=V3_MIN_PROVIDER_POOL_MICRO,
            burn_bps=V3_BURN_BPS,
            protocol_bps=V3_PROTOCOL_BPS,
            provider_bps=V3_PROVIDER_BPS,
        )
        reward_micro = scheduled_work_reward_micro(inference_index)
        return {
            "version": version,
            "metering": metering,
            "pricing": pricing,
            "emission": {
                "schedule": "verified-inference-halving-v1",
                "inferenceIndex": inference_index,
                "halvingIndex": (
                    inference_index // V3_WORK_REWARD_HALVING_INFERENCES
                ),
                "halvingIntervalInferences": (
                    V3_WORK_REWARD_HALVING_INFERENCES
                ),
                "blockRewardMicro": reward_micro,
                "requestedIssuanceMicro": reward_micro,
            },
        }

    if version != LEGACY_ECONOMICS_VERSION:
        raise ValueError("unsupported CTX economics version")

    pricing = _legacy_pricing(
        provider_work=provider_work,
        provider_rates_micro=provider_rates_micro,
        minimum_provider_pool_micro=V2_MIN_PROVIDER_POOL_MICRO,
        burn_bps=V2_BURN_BPS,
        protocol_bps=V2_PROTOCOL_BPS,
        provider_bps=V2_PROVIDER_BPS,
    )
    epoch = inference_index // INFERENCES_PER_EPOCH
    epoch_offset = inference_index % INFERENCES_PER_EPOCH
    epoch_budget_micro = scheduled_epoch_emission_micro(epoch)
    if target_capacity_units <= 0:
        raise ValueError("capacity target must be positive")
    ready_capacity_units = max(0, min(ready_capacity_units, target_capacity_units))
    capacity_unlock_bps = ready_capacity_units * 10_000 // target_capacity_units
    demand_unlock_bps = min(
        10_000,
        max(0, epoch_fees_before_micro + pricing["feeMicro"])
        * 10_000
        // EPOCH_DEMAND_TARGET_MICRO,
    )
    subsidy_ceiling_micro = (
        pricing["providerPoolMicro"]
        * MAX_SUBSIDY_BPS_OF_PROVIDER_FEES
        // 10_000
    )
    requested_issuance_micro = (
        subsidy_ceiling_micro
        * capacity_unlock_bps
        // 10_000
        * demand_unlock_bps
        // 10_000
    )
    epoch_remaining_micro = max(
        0, epoch_budget_micro - epoch_issuance_before_micro
    )
    return {
        "version": version,
        "metering": metering,
        "pricing": pricing,
        "emission": {
            "epoch": epoch,
            "epochOffset": epoch_offset,
            "epochBudgetMicro": epoch_budget_micro,
            "epochFeesBeforeMicro": max(0, epoch_fees_before_micro),
            "epochIssuanceBeforeMicro": max(0, epoch_issuance_before_micro),
            "epochRemainingMicro": epoch_remaining_micro,
            "readyCapacityUnits": ready_capacity_units,
            "targetCapacityUnits": target_capacity_units,
            "capacityUnlockBps": capacity_unlock_bps,
            "demandTargetMicro": EPOCH_DEMAND_TARGET_MICRO,
            "demandUnlockBps": demand_unlock_bps,
            "maxSubsidyBpsOfProviderFees": MAX_SUBSIDY_BPS_OF_PROVIDER_FEES,
            "requestedIssuanceMicro": requested_issuance_micro,
        },
    }


def public_policy() -> dict:
    return {
        "version": ECONOMICS_VERSION,
        "reward": {
            "rate": tokens(REWARD_RATE_MICRO_PER_WORK_UNIT),
            "rateMicroPerWorkUnit": REWARD_RATE_MICRO_PER_WORK_UNIT,
            "unit": "verified model-compute-unit × weighted-token",
            "eligibility": "signed work in a finalized inference block",
            "distribution": "each provider is paid for its own verified work",
        },
    }
