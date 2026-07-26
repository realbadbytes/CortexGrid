# CTX rewards

CortexGrid uses one reward rule:

```text
worker reward = verified work units × 0.0001 CTX
```

A work unit is one model compute unit applied to one weighted token. Output
tokens count twice because generation is more expensive than prompt processing:

```text
weighted tokens = input tokens + 2 × output tokens
work units      = assigned model compute units × weighted tokens
```

After an inference completes, every selected worker signs a receipt. The
controller verifies those receipts and mints the calculated CTX directly to
each worker's reward address. More verified processing earns more CTX. Failed,
invalid, unselected, offline, and idle machines earn nothing.

CortexGrid Chat participants execute their assigned stage on their own
successful prompts. Dedicated workers are assigned work from network requests.
Both roles use exactly the same reward calculation.

The reward calculation has:

- no lottery or block jackpot;
- no provider bidding;
- no inference fee paid in CTX;
- no fee split or token removal;
- no epochs or halvings.

End-user subscriptions, API charges, or other service billing are separate from
the CTX ledger. The ledger only records verified work, rewards, transfers,
balances, and the 1 billion CTX maximum supply.

The devnet initially allocated 1 million CTX. New CTX can only be created by
verified inference work, and the final reward is clipped if it would exceed the
maximum supply. Existing v2 and v3 devnet blocks remain replayable.
