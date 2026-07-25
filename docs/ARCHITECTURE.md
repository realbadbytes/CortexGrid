# Architecture and production boundary

## Devnet data flow

```text
1. A wallet-free end user submits a prompt through CortexTerm or the HTTP API
2. Python selects one online replica for every ordered model stage
3. A persistent llama.cpp gateway loads tensors across the selected RPC route
4. The real SmolLM2 model performs prefill and autoregressive generation
5. Both participant identities sign chained participation attestations
6. Python verifies every signature and commitment link
7. Receipt digests form a Merkle root
8. One useful-work block settles the metered gateway fee, burn, rewards, and
   capacity-and-demand-limited issuance
```

The chain accepts an inference block only when:

- every receipt has a valid Ed25519 signature;
- receipts form a continuous activation chain;
- stages are complete and ordered;
- request, model, and output commitments match the block;
- reward recipients are derived from receipt public keys;
- the fee split and work-weighted rewards are exact;
- capped issuance does not exceed the maximum supply;
- metered prices, capacity/demand unlocks, and epoch limits recompute exactly;
- balances conserve the resulting total supply.

Transfers use Ed25519 signatures and per-wallet nonces. They cannot invoke the
issuance transaction type.

## Why Python plus C++/CUDA

Python is not in the tensor kernel. It performs control work that is easy to
inspect: placement, routing, process lifecycle, verification, settlement, and
HTTP. llama.cpp owns model tensors, KV cache, tokenization, sampling, and
C++/CUDA computation.

The real-model path uses llama.cpp behind a controller-owned mTLS gateway:

```text
Python coordinator
       │ sole client certificate
       ▼
llama.cpp gateway
       ├════ TLS 1.3 + provider pin ════▶ TLS tunnel A → loopback RPC A
       └════ TLS 1.3 + provider pin ════▶ TLS tunnel B → loopback RPC B
```

Every raw RPC server binds to loopback. A provider exposes only an mTLS tunnel
that requires a CA-valid certificate and the exact controller client
certificate fingerprint. The controller pins the provider certificate before
placing it in a route. A network member has a server-only certificate and
cannot invoke another member's RPC endpoint.

```text
Provider                 Coordinator
   │ signed CSR              │
   ├────────────────────────▶│
   │ server-only certificate │
   │◀────────────────────────┤
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

## Five TiB capacity profile

The built-in `5tb` profile is an exact 5 TiB manifest: 160 stages contain 32 GiB
of weights each and reserve 40 GiB each. One complete route therefore requires
160 compatible providers and 6.25 TiB of aggregate RAM. Target replication
requires 320 placements and 12.5 TiB of aggregate RAM. Placement always chooses
the compatible stage with the fewest replicas; model readiness requires at
least one online replica for every stage.

The profile's artifact descriptors are deliberately marked unavailable. They
prove that assignment, byte accounting, commitments, coverage, and failure
states scale to the target without claiming that a 5 TiB checkpoint exists in
this repository.

## Production sequence

Keep the same three boundaries and deepen them in this order:

1. Add certificate expiry automation, revocation, and controller key rotation.
2. Add direct worker-to-worker activation streams with per-job authorization.
3. Add redundant execution sampling or trusted/zero-knowledge execution so signatures
   prove correctness, not only node authorship.
4. Replicate the Python state machine behind a Byzantine-finality protocol.
5. Add peer discovery, mempool rules, snapshots, pruning, and safe key custody.
6. Commission independent protocol, cryptographic, token-economic, and legal
   reviews before any public token sale or market listing.

The devnet is intentionally not split into microservices. New infrastructure
should be introduced only when a measured production constraint requires it.
