# Native executor protocol

The Python coordinator keeps one `cortex_worker` process alive for each logical
node. Communication uses UTF-8, tab-delimited lines over stdin/stdout.

This protocol is intentionally tiny. It is a local development transport that
will later map directly to a peer RPC without changing receipt semantics.

## Handshake

The coordinator supplies a 32-byte Ed25519 private key in the worker process
environment and starts:

```text
cortex_worker --serve --node-id atlas
```

The worker responds:

```text
READY  cg-executor-v1  cpu  PUBLIC_KEY_HEX
```

Fields above are separated by tabs. A CUDA build reports
`cuda,cpu-fallback`.

## Execute

The coordinator sends nine fields:

```text
EXEC  REQUEST_ID  MODEL_ID  VERSION  SHARD_INDEX  INPUT_DIGEST  WEIGHTS_DIGEST  COMPUTE_UNITS  CHALLENGE
```

The worker responds:

```text
RESULT  REQUEST_ID  OUTPUT_DIGEST  BACKEND  KERNEL_MICROS  SIGNATURE
```

The Ed25519 signature covers:

```text
request|model|version|shard|node|input|output|weights|work|challenge|backend|micros
```

This binds a receipt to one node, one model revision, one weight commitment,
one activation edge, and one coordinator challenge.

## Attest model RPC participation

The real-model path asks the same node identity to sign the commitments around
work performed by its private llama.cpp RPC backend:

```text
ATTEST  REQUEST  MODEL  VERSION  STAGE  INPUT  OUTPUT  WEIGHTS  WORK  CHALLENGE  BACKEND  MICROS
```

The worker returns:

```text
ATTESTED  REQUEST  SIGNATURE
```

This is deliberately called a participation attestation. It binds the node to
the request and result, but it does not independently prove floating-point
correctness. Random redundant execution is the next verification layer.

## Shutdown

```text
STOP
```

The worker responds with `STOPPED` and exits.

Private keys never appear in a protocol frame or HTTP response. In production,
each remote provider—not the coordinator—owns its node key.

## Remote provider join protocol

Remote providers use `cortex-join-v1`:

1. `POST /api/nodes/register` with capabilities, RPC/control endpoints,
   micro-CTX/work-unit bid, timestamp, nonce, public key, and an Ed25519
   signature.
2. The coordinator challenges `GET /v1/status` to prove that the registering key
   controls the advertised endpoint.
3. The coordinator returns a signed model/stage assignment.
4. The provider persists and verifies the assignment, then signs
   `POST /api/nodes/{id}/ready`.
5. Signed heartbeats renew a 45-second routing lease and return the current CTX
   balance.

Every `/v1/execute` or `/v1/attest` request carries a unique challenge and a
coordinator-signed work ticket bound to the provider, assignment, model,
stage, input, and weight commitment. Providers reject invalid, mismatched, or
replayed tickets. The signed assignment binds the reward address and fixed
reward policy.
Inference receipts remain signed by the provider's own key, and rewards are
paid to the CTX address derived from that key.

Before real inference starts, the coordinator sends `/v1/jobs/start`. The
provider exposes that signed job in its active-work view. Attestation moves it
to pending settlement, and `/v1/jobs/settle` records the final block, proof
root, CTX reward, and balance. Failed generations receive `/v1/jobs/cancel`.

The provider-local operator API exposes:

- `GET /v1/provider` — identity, health, assignment, model/layers, jobs, balance;
- `GET /v1/jobs` — active and recent jobs;
- `POST /v1/operator/leave` — sign a coordinator leave request;
- `POST /v1/operator/join` — register and become ready again.

These operations require a bearer token stored separately from the provider's
Ed25519 key. Coordinator-facing work endpoints still require signed work
tickets, not the operator token.
