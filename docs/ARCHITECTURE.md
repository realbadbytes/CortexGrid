# Architecture and production boundary

## Devnet data flow

```text
1. CortexGrid Chat signs a prompt through an online interactive compute node
2. Python selects one online replica for each of two model stages
3. A persistent llama.cpp gateway loads tensors across both RPC participants
4. The real SmolLM2 model performs prefill and autoregressive generation
5. Both participant identities sign chained participation attestations
6. Python verifies every signature and commitment link
7. Receipt digests form a Merkle root
8. The gateway pays the CTX service fee, and one useful-work block settles
   provider revenue plus declining bootstrap issuance
```

The chain accepts an inference block only when:

- every receipt has a valid Ed25519 signature;
- receipts form a continuous activation chain;
- stages are complete and ordered;
- request, model, and output commitments match the block;
- reward recipients are derived from receipt public keys;
- provider quotes equal verified work multiplied by signed service rates;
- service payment is conserved between providers and the coordinator;
- bootstrap issuance matches only paid provider service revenue;
- capped issuance does not exceed the maximum supply;
- metered work and rewards recompute exactly;
- balances conserve the resulting total supply.

Transfers use Ed25519 signatures and per-wallet nonces. They cannot invoke the
issuance transaction type. When bootstrap issuance ends, inference settlement
continues entirely through gateway-funded service payments.

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

Built-in RPC servers bind to loopback. Independently operated providers can run
the node daemon across a trusted private network. The daemon owns its key,
registers its capabilities and endpoints, verifies its signed stage assignment,
heartbeats, and accepts only coordinator-signed work tickets.

```text
Provider                 Coordinator
   │ REGISTER + signature    │
   ├────────────────────────▶│
   │ signed stage assignment │
   │◀────────────────────────┤
   │ READY + signature       │
   ├────────────────────────▶│
   │ heartbeat / balance     │
   ├────────────────────────▶│
```

When selected, llama.cpp transfers the assigned layer tensors into
provider-owned RPC buffers in RAM/VRAM; its optional cache can persist large
tensors. The provider signs its own inference receipt, so CTX rewards are
derived from the provider's public key rather than a coordinator-owned
identity.

## Production sequence

Keep the same three boundaries and deepen them in this order:

1. Replace llama.cpp's unauthenticated tensor RPC with authenticated transport.
2. Add direct worker-to-worker activation streams.
3. Add redundant execution sampling or trusted/zero-knowledge execution so signatures
   prove correctness, not only node authorship.
4. Replicate the Python state machine behind a Byzantine-finality protocol.
5. Add peer discovery, mempool rules, snapshots, pruning, and safe key custody.
6. Commission independent protocol, cryptographic, token-economic, and legal
   reviews before any public token sale or market listing.

The devnet is intentionally not split into microservices. New infrastructure
should be introduced only when a measured production constraint requires it.
