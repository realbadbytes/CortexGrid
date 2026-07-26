# CTX service economics

CTX v5 ties provider revenue to paid inference demand.

```text
weighted tokens = input tokens + 2 × output tokens
work units      = model compute units × weighted tokens
provider quote  = work units × provider price
service fee     = provider quotes ÷ 95%
provider reward = provider quote + bootstrap issuance
```

## Who pays

The service gateway sponsors the on-chain CTX payment, so an end user does not
need a wallet or CTX. A production gateway can charge that user through a
subscription, fiat invoice, or another product-level payment method, then
replenish its CTX balance. Every inference therefore creates actual CTX usage
without exposing token mechanics to the user.

The payment is atomic with the finalized inference block:

- 95% pays the selected providers at their advertised rates;
- 5% goes to the coordinator treasury for network operation;
- 0% is burned.

Providers choose a price in micro-CTX per verified work unit. The coordinator
routes the lowest-priced compatible replica first, then considers completed
jobs and latency. The sponsored devnet accepts rates up to 100 micro-CTX per
work unit. A provider must choose a rate that covers its hardware, electricity,
bandwidth, and operating costs.

## Bootstrap issuance

Paid service demand—not idle capacity—unlocks new CTX. At launch, issuance
matches 200% of provider service payments:

```text
bootstrap issuance = provider service payments × current match
initial match      = 2.00×
halving interval   = 1,000,000 finalized inferences
```

The match becomes 1.00× after one million inferences, 0.50× after two million,
and continues halving. Issuance is clipped at the 1 billion CTX maximum supply.
Provider service payments continue unchanged when the match reaches zero or
the supply cap is reached.

This separates two jobs:

- service payments make ongoing provider operation economically possible;
- temporary issuance compensates early providers for joining a smaller,
  less-utilized network.

## Settlement guarantees

Every selected provider signs a receipt. The controller checks the route,
model and activation commitments, metered work, advertised prices, service
payment, issuance match, recipients, and supply limit. Invalid, failed,
offline, idle, or unselected machines receive nothing.

Transfers never create supply. v2, v3, and v4 blocks remain replayable, but all
new blocks use `ctx-economics-v5`.

CTX market price and provider profitability remain external outcomes. The
protocol supplies measurable work, paid demand, and deterministic settlement;
it cannot guarantee exchange liquidity or a fiat return.
