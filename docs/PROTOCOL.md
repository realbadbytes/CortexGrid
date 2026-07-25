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
each remote provider—not the coordinator—must own its node key.
