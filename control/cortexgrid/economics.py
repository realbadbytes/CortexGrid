from __future__ import annotations


TOKEN_UNIT = 1_000_000
TOKEN_SYMBOL = "CTX"
MAX_SUPPLY_MICRO = 1_000_000_000 * TOKEN_UNIT
GENESIS_SUPPLY_MICRO = 1_000_000 * TOKEN_UNIT

ECONOMICS_VERSION = "ctx-economics-v2"
PRICE_MIN_MICRO_PER_WORK_UNIT = 1
PRICE_MAX_MICRO_PER_WORK_UNIT = 10_000
DEFAULT_PRICE_MICRO_PER_WORK_UNIT = 10

PREFILL_WEIGHT = 1
DECODE_WEIGHT = 2
MIN_PROVIDER_POOL_MICRO = 50_000
BURN_BPS = 500
PROTOCOL_BPS = 1_000
PROVIDER_BPS = 10_000 - BURN_BPS - PROTOCOL_BPS

INFERENCES_PER_EPOCH = 1_000
EMISSION_HALVING_EPOCHS = 100
INITIAL_EPOCH_EMISSION_MICRO = 100 * TOKEN_UNIT
MAX_SUBSIDY_BPS_OF_PROVIDER_FEES = 2_000
EPOCH_DEMAND_TARGET_MICRO = 100 * TOKEN_UNIT
CAPACITY_TARGET_REPLICAS = 2


def tokens(amount_micro: int) -> float:
    return round(amount_micro / TOKEN_UNIT, 6)


def ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_div requires a non-negative numerator and denominator")
    return (numerator + denominator - 1) // denominator


def scheduled_epoch_emission_micro(epoch: int) -> int:
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    halvings = epoch // EMISSION_HALVING_EPOCHS
    return INITIAL_EPOCH_EMISSION_MICRO >> halvings


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
) -> dict:
    if inference_index < 0:
        raise ValueError("inference index cannot be negative")
    if kernel_micros < 0:
        raise ValueError("kernel time cannot be negative")
    if set(provider_rates_micro) != set(provider_compute_units):
        raise ValueError("every participating provider needs exactly one price")
    if sum(provider_compute_units.values()) != model_compute_units:
        raise ValueError("provider work does not equal model work")
    normalized_rates = {
        address: validate_price(price)
        for address, price in sorted(provider_rates_micro.items())
    }
    provider_work = {
        address: metered_work_units(
            int(provider_compute_units[address]), input_tokens, output_tokens
        )
        for address in sorted(provider_compute_units)
    }
    quoted_provider_cost_micro = sum(
        provider_work[address] * normalized_rates[address]
        for address in provider_work
    )
    target_provider_pool_micro = max(
        MIN_PROVIDER_POOL_MICRO, quoted_provider_cost_micro
    )
    fee_micro = ceil_div(target_provider_pool_micro * 10_000, PROVIDER_BPS)
    burn_micro = fee_micro * BURN_BPS // 10_000
    protocol_micro = fee_micro * PROTOCOL_BPS // 10_000
    provider_pool_micro = fee_micro - burn_micro - protocol_micro

    epoch = inference_index // INFERENCES_PER_EPOCH
    epoch_offset = inference_index % INFERENCES_PER_EPOCH
    epoch_budget_micro = scheduled_epoch_emission_micro(epoch)
    if target_capacity_units <= 0:
        raise ValueError("capacity target must be positive")
    ready_capacity_units = max(0, min(ready_capacity_units, target_capacity_units))
    capacity_unlock_bps = ready_capacity_units * 10_000 // target_capacity_units
    demand_unlock_bps = min(
        10_000,
        max(0, epoch_fees_before_micro + fee_micro)
        * 10_000
        // EPOCH_DEMAND_TARGET_MICRO,
    )
    subsidy_ceiling_micro = (
        provider_pool_micro * MAX_SUBSIDY_BPS_OF_PROVIDER_FEES // 10_000
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
        "version": ECONOMICS_VERSION,
        "metering": {
            "modelComputeUnits": model_compute_units,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "kernelMicros": kernel_micros,
            "prefillWeight": PREFILL_WEIGHT,
            "decodeWeight": DECODE_WEIGHT,
            "meteredWorkUnits": sum(provider_work.values()),
            "providerWorkUnits": provider_work,
        },
        "pricing": {
            "providerRatesMicroPerWorkUnit": normalized_rates,
            "quotedProviderCostMicro": quoted_provider_cost_micro,
            "minimumProviderPoolMicro": MIN_PROVIDER_POOL_MICRO,
            "feeMicro": fee_micro,
            "burnBps": BURN_BPS,
            "protocolBps": PROTOCOL_BPS,
            "providerBps": PROVIDER_BPS,
            "burnMicro": burn_micro,
            "protocolMicro": protocol_micro,
            "providerPoolMicro": provider_pool_micro,
        },
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
        "pricing": {
            "unit": "model-compute-unit × weighted-token",
            "defaultProviderPriceMicroPerWorkUnit": (
                DEFAULT_PRICE_MICRO_PER_WORK_UNIT
            ),
            "minimumProviderPool": tokens(MIN_PROVIDER_POOL_MICRO),
            "burnPercent": BURN_BPS / 100,
            "protocolPercent": PROTOCOL_BPS / 100,
            "providerPercent": PROVIDER_BPS / 100,
            "prefillWeight": PREFILL_WEIGHT,
            "decodeWeight": DECODE_WEIGHT,
        },
        "emission": {
            "inferencesPerEpoch": INFERENCES_PER_EPOCH,
            "initialEpochBudget": tokens(INITIAL_EPOCH_EMISSION_MICRO),
            "halvingEpochs": EMISSION_HALVING_EPOCHS,
            "demandTargetPerEpoch": tokens(EPOCH_DEMAND_TARGET_MICRO),
            "capacityTargetReplicas": CAPACITY_TARGET_REPLICAS,
            "maxSubsidyPercentOfProviderFees": (
                MAX_SUBSIDY_BPS_OF_PROVIDER_FEES / 100
            ),
        },
    }
