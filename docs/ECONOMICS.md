# CTX economics v2

CTX pays for useful inference. It does not promise a market price or provider
profit.

## Service pricing

Users do not need wallets. A gateway funds each request and can recover that
cost through subscriptions, API billing, enterprise contracts, or another
external revenue source.

Each provider advertises a bid in micro-CTX per work unit. For one provider:

```text
weighted tokens = input tokens + 2 × output tokens
work units      = assigned model compute units × weighted tokens
quote           = work units × provider bid
```

The gateway selects lower-priced ready replicas first. The inference fee is the
smallest integer fee whose 85% provider share covers all selected quotes, with a
minimum provider pool of 0.05 CTX. Settlement sends:

- 85% to participating providers, weighted by their quotes;
- 10% to the protocol treasury;
- 5% to permanent burn.

Every provider receives at least its quoted amount. Model compute units are
protocol-defined and must eventually be calibrated from architecture FLOPs and
memory traffic rather than provider-reported runtime.

## Issuance

The maximum supply remains 1 billion CTX. Genesis creates 1 million CTX. New
CTX can be issued only inside a valid inference block.

There are 1,000 finalized inferences per economic epoch. The initial epoch
issuance ceiling is 100 CTX and halves every 100 epochs:

```text
epoch budget = 100 CTX / 2^(floor(epoch / 100))
```

For an eligible inference:

```text
capacity unlock = min(1, ready assigned capacity / two-replica target)
demand unlock   = min(1, epoch service fees / 100 CTX)
subsidy ceiling = 20% × provider fee pool

requested issuance =
    subsidy ceiling × capacity unlock × demand unlock
```

Actual issuance is the smallest of requested issuance, remaining epoch budget,
and remaining maximum-supply headroom. It is distributed by verified metered
work. Capacity unlocks the subsidy but never receives it while idle. Unused
epoch budget remains unminted.

Fee burn and issuance are separate, so supply can be inflationary or
deflationary depending on demonstrated network demand. All inputs, rates,
unlocks, transactions, and epoch counters are committed in the inference block
and deterministically revalidated with the chain.

## Security boundary

The current devnet verifies signatures, assignments, liveness, model
commitments, metering arithmetic, supply conservation, and issuance limits.
Provider bids and capacity are coordinator-attested network state, and token
counts are reported by the model runtime. A public mainnet still needs:

1. an on-chain provider registry with signed bids, bonds, and slashing;
2. independent capacity benchmarks and availability challenges;
3. authenticated metering or replicated correctness checks;
4. paid gateway admission and rate limits to prevent sponsored traffic farming;
5. Byzantine block finality and independent economic/security review.
