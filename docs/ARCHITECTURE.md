# Architecture and production boundary

## Devnet data flow

```text
1. Client submits prompt and pays 1 CTX
2. Python selects one online replica for each of two model stages
3. A persistent llama.cpp gateway loads tensors across both RPC participants
4. The real SmolLM2 model performs prefill and autoregressive generation
5. Both participant identities sign chained participation attestations
6. Python verifies every signature and commitment link
7. Receipt digests form a Merkle root
8. One useful-work block settles fee, burn, rewards, and issuance
```

The chain accepts an inference block only when:

- every receipt has a valid Ed25519 signature;
- receipts form a continuous activation chain;
- stages are complete and ordered;
- request, model, and output commitments match the block;
- reward recipients are derived from receipt public keys;
- the fee split and work-weighted rewards are exact;
- capped issuance does not exceed the maximum supply;
- balances conserve the resulting total supply.

Transfers use Ed25519 signatures and per-wallet nonces. They cannot invoke the
issuance transaction type.

## Why Python plus C++/CUDA

Python is not in the tensor kernel. It performs control work that is easy to
inspect: placement, routing, process lifecycle, verification, settlement, and
HTTP. llama.cpp owns model tensors, KV cache, tokenization, sampling, and
C++/CUDA computation.

The first real-model milestone uses llama.cpp's private RPC gateway:

```text
Python coordinator
       │ route
       ▼
llama.cpp gateway
       ├════ remote tensor work ════▶ RPC participant A
       └════ remote tensor work ════▶ RPC participant B
```

The RPC servers bind only to loopback because upstream documents that transport
as a proof of concept without authentication. Production replaces this star
transport with the same logical stage interface over authenticated,
worker-to-worker streams.

## Production sequence

Keep the same three boundaries and deepen them in this order:

1. Wrap the model backend and signer into one remotely deployable node daemon.
2. Add direct authenticated peer transport and route tickets.
3. Add redundant execution sampling or trusted/zero-knowledge execution so signatures
   prove correctness, not only node authorship.
4. Replicate the Python state machine behind a Byzantine-finality protocol.
5. Add peer discovery, mempool rules, snapshots, pruning, and safe key custody.
6. Commission independent protocol, cryptographic, token-economic, and legal
   reviews before any public token sale or market listing.

The devnet is intentionally not split into microservices. New infrastructure
should be introduced only when a measured production constraint requires it.
