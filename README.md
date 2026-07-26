# CortexGrid

> **CortexGrid lets people pool RAM, VRAM, and compute to run language models
> that would be too large or expensive for any participant to run alone.**
> Every machine increases model capacity or throughput. Every machine that
> completes verified inference work earns CTX.

CortexGrid uses a simple centralized coordinator: Python assigns model stages,
routes inference, verifies work, and finalizes the CTX ledger. C++/CUDA and
llama.cpp perform model execution on participant hardware.

```text
                         signed stage assignments
                    ┌────────── Coordinator ──────────┐
model weights ─────▶│ layers 0–14  ─▶ provider A     │
                    │ layers 15–29 ─▶ provider B     │
                    │ replicas      ─▶ provider C…    │
                    └─────────────────────────────────┘
                                      │
CortexGrid Chat ─▶ distributed inference ─▶ signed proof ─▶ CTX rewards
```

## Start the network

Requirements: Python 3.11+, CMake 3.20+, a C++20 compiler, OpenSSL development
headers, and Python `cryptography`. CUDA is optional.

```bash
make bootstrap-model
make run-real
```

Open [http://localhost:65534](http://localhost:65534). The dashboard shows live
network capacity and inference activity, then guides a new machine through
joining CortexGrid. `COORDINATOR_PUBLIC_KEY` is the `coordinatorPublicKey`
reported by `/api/health`; confirm it through a trusted channel before joining
a remote network.

## Join, chat, and earn CTX

This mode opens CortexGrid Chat while also offering the machine's compute to
the network:

```bash
make bootstrap-provider
make chat-browser \
  COORDINATOR_URL=http://COORDINATOR_IP:65534 \
  COORDINATOR_PUBLIC_KEY=OUT_OF_BAND_KEY \
  ADVERTISE_HOST=THIS_MACHINE_IP \
  TERM_MEMORY_GIB=24
```

The command creates a local provider identity and reward wallet, receives a
model stage, starts the local compute bridge, and opens the browser. No CTX or
pre-existing wallet is required. Closing the process leaves the network.

For the terminal interface:

```bash
make chat \
  COORDINATOR_URL=http://COORDINATOR_IP:65534 \
  COORDINATOR_PUBLIC_KEY=OUT_OF_BAND_KEY \
  ADVERTISE_HOST=THIS_MACHINE_IP \
  TERM_MEMORY_GIB=24
```

Useful commands are `/status`, `/route`, `/proof`, `/new`, `/help`, and
`/exit`. `/status` shows the hosted stage and current CTX balance.

## Join as a dedicated worker

A dedicated worker serves inference without opening a chat client:

```bash
make bootstrap-provider
make join \
  NODE_ID=my-gpu-01 \
  NODE_NAME="My GPU" \
  MEMORY_GIB=24 \
  REGION=us-east \
  COORDINATOR_URL=http://COORDINATOR_IP:65534 \
  COORDINATOR_PUBLIC_KEY=OUT_OF_BAND_KEY \
  ADVERTISE_HOST=THIS_MACHINE_IP
```

Operate it from another terminal:

```bash
./CortexNode status     --node-id my-gpu-01
./CortexNode jobs       --node-id my-gpu-01
./CortexNode balance    --node-id my-gpu-01
./CortexNode assignment --node-id my-gpu-01
./CortexNode leave      --node-id my-gpu-01
./CortexNode join       --node-id my-gpu-01
```

## Model placement

The model manifest divides weights into ordered layer stages. When a machine
registers its available memory and compute, the coordinator assigns a stage
that fits. The signed assignment identifies the model version, layer range,
memory requirement, and weights digest. llama.cpp transfers those tensors into
the provider's RAM or VRAM and reuses its local cache.

Inference runs through one provider for each stage. More aggregate memory lets
the network place larger models; more replicas and compute increase throughput
and fault tolerance.

## CTX

CTX is created only for completed, verified inference:

```text
provider reward = verified work units × 0.0001 CTX
```

Every selected provider signs a work receipt. The coordinator verifies the
route, model commitments, output commitments, signatures, and reward amount
before finalizing a block and crediting provider reward addresses. Idle,
offline, invalid, and unselected machines mint nothing. Supply is capped at
1 billion CTX.

Balances are stored on the CTX ledger. The key controlling a worker's address
stays in `.provider/NODE_ID/node_wallet.json`; the default chat identity uses
`.provider/cortexterm/node_wallet.json` for compatibility.

Network APIs:

```bash
curl http://COORDINATOR_IP:65534/api/network
curl http://COORDINATOR_IP:65534/api/models
curl http://COORDINATOR_IP:65534/api/chain
curl http://COORDINATOR_IP:65534/api/economics
```

See [CTX economics](docs/ECONOMICS.md) and the
[protocol](docs/PROTOCOL.md) for ledger and proof details.

## Verify

```bash
make test
make test-real
```

The included real-model demo uses SmolLM2 135M. The manifest and placement
architecture can describe much larger models, but the devnet is not yet a
production-scale multi-terabyte deployment. Registration, assignments, work
tickets, and receipts are signed. The upstream llama.cpp tensor RPC must still
run on a trusted private network or VPN. Work signatures prove attributable
participation, not Byzantine-safe numerical correctness. CTX has no guaranteed
market value or exchange listing.
