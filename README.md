# CortexGrid

CortexGrid is a devnet for distributed AI inference with a native utility
token. Python coordinates inference and consensus; llama.cpp runs model layers
on native C++/CUDA participants.

```text
Client → Python coordinator → llama.cpp gateway
                              ├─ Atlas: layers 0–14
                              └─ Ember: layers 15–29
                    → signed attestations → CTX block and rewards
```

The included SmolLM2 135M Q4_K_M model is split across two private RPC
participants. Each stage has three candidate replicas, so an unavailable node
can be replaced. Successful participants sign Ed25519 attestations, which are
committed to a Merkle root and verified before the inference fee, burn,
issuance, and worker rewards are finalized in an append-only CTX block.

## Run

Requirements: Python 3.11+, CMake 3.20+, a C++20 compiler, OpenSSL development
headers, and Python `cryptography`. CUDA and an NVIDIA GPU are optional.

```bash
make bootstrap-model
make run-real
```

Open <http://localhost:4173>, or submit an inference directly:

```bash
curl http://localhost:4173/api/inference \
  -H 'content-type: application/json' \
  -d '{"prompt":"What is distributed inference? Answer in one sentence."}'
```

Inspect the chain with:

```bash
curl http://localhost:4173/api/chain
curl http://localhost:4173/api/blocks/1
```

## Join the inference network

The current devnet launches six local participants: `atlas`, `ember`, `kite`,
`orbit`, `sable`, and `nova`. Make a participant available for routing with:

```bash
curl http://localhost:4173/api/nodes/atlas/status \
  -H 'content-type: application/json' \
  -d '{"status":"online"}'
```

Submit an inference, then check which participants were selected in the
response's `route`. A selected participant is paid only after its signed
attestation verifies and the CTX block is finalized.

List each participant's reward address, completed jobs, and CTX balance:

```bash
curl -sS http://localhost:4173/api/nodes | python3 -c '
import json, sys
for node in json.load(sys.stdin)["nodes"]:
    print(node["id"], node["rewardAddress"],
          "jobs:", node["completedJobs"], "CTX:", node["balance"])
'
```

The same balances are recorded in `GET /api/chain`. Arbitrary remote hosts
cannot register yet; joining this version means running one of the coordinator's
predefined local participants. Permissionless remote joining requires the
authenticated node daemon described under **Current boundary**.

## Verify

```bash
make test
make test-real
```

The real-model suite verifies two-node generation, equality with a
deterministic one-node reference, valid CTX settlement, and replica failover.
The build uses CUDA automatically when `nvcc` is available and otherwise uses
llama.cpp's CPU backend.

Pinned artifacts:

- llama.cpp commit: `720d7fa4097f76e5d0eade5a92c1df87c1faf9d9`
- Model: `SmolLM2-135M-Instruct-Q4_K_M.gguf`
- Model SHA-256:
  `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`

## Current boundary

This is a local-process devnet. llama.cpp RPC is restricted to loopback because
its upstream transport is experimental and unauthenticated. Worker signatures
prove participation and make token settlement auditable; they do not yet prove
the numerical correctness of every model operation. Production deployment
needs authenticated remote node transport, replicated correctness checks, and
Byzantine block finality.

CTX has protocol utility, but utility cannot guarantee market price or monetary
value. A public launch requires independent security, economic, and legal
review. See [the architecture notes](docs/ARCHITECTURE.md) and
[worker protocol](docs/PROTOCOL.md) for details.
