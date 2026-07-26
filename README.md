# CortexGrid

> **Goal:** let everyone who contributes compute to CortexGrid run language
> models that would be financially or technically out of reach on one machine.
> More participating RAM, VRAM, and compute means the network can host larger
> models, keep more replicas, and serve more inference concurrently.

CortexGrid pools member hardware into a distributed inference service. Model
stages run across native C++/CUDA workers; Python coordinates placement and
routing. Its proof of work is useful model inference—not arbitrary hashing—and
finalized work earns workers the CTX cryptocurrency.

```text
                       signed model assignments
Coordinator ──────────────────────────────────────────────┐
    │                                                     │
    ├── stage 0 / layers 0–14 ── participant A (RAM/GPU) │
    ├── stage 1 / layers 15–29 ─ participant B (RAM/GPU) │
    └── replicas / throughput ─── dedicated workers       │
                                                          │
CortexGrid Chat ─── distributed inference ── proof ── CTX block/rewards
```

## Two ways to participate

1. **Use and serve:** CortexGrid Chat joins as a compute provider, receives a
   model stage, executes that stage on its own prompts, serves other network
   jobs, earns CTX for successful work, and lets you chat with the pooled
   model. Inference access lasts only while your node is online.
2. **Serve only:** a dedicated worker stays online to host model stages and run
   inference for others, earning CTX without submitting its own prompts.

Neither role earns CTX merely for being idle. Rewards require selected,
verified, finalized inference work.

## Run a coordinator

Requirements: Python 3.11+, CMake 3.20+, a C++20 compiler, OpenSSL development
headers, and Python `cryptography`. CUDA is optional.

```bash
make bootstrap-model
make run-real
```

The API and dashboard listen at `http://localhost:65534`. Providers should pin
the `coordinatorPublicKey` returned by `GET /api/health`.

## Use and serve with CortexGrid Chat

First build the provider runtime:

```bash
make bootstrap-provider
```

For the browser interface:

```bash
make chat-browser \
  COORDINATOR_URL=http://COORDINATOR_IP:65534 \
  COORDINATOR_PUBLIC_KEY=OUT_OF_BAND_KEY \
  ADVERTISE_HOST=THIS_MACHINE_IP \
  TERM_MEMORY_GIB=24
```

This starts the local compute bridge and opens CortexGrid Chat in the browser.
The bridge token stays in the URL fragment and is never sent to the
coordinator. Closing the bridge leaves the network and stops serving.

For the terminal interface:

```bash
make chat \
  COORDINATOR_URL=http://COORDINATOR_IP:65534 \
  COORDINATOR_PUBLIC_KEY=OUT_OF_BAND_KEY \
  ADVERTISE_HOST=THIS_MACHINE_IP \
  TERM_MEMORY_GIB=24
```

CortexGrid Chat creates a provider identity and reward wallet, joins, serves
its assigned stage, and signs each prompt. You need no existing CTX and no
manually configured wallet. `/status` shows your stage and CTX balance;
`/exit` leaves the network.

Useful commands: `/status`, `/route`, `/proof`, `/new`, `/help`, and `/exit`.
For one prompt:

```bash
./CortexChat --server http://COORDINATOR_IP:65534 \
  --coordinator-public-key OUT_OF_BAND_KEY \
  --advertise-host THIS_MACHINE_IP \
  --memory-gib 24 \
  -p "Explain pipeline parallelism."
```

Your assigned stage is routed through your node for your own successful prompt,
so you earn a share of that inference's CTX reward. Your node remains eligible
for other routed jobs while CortexGrid Chat is online.

## Run a dedicated worker

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

Inspect or operate it from another terminal:

```bash
./CortexNode status     --node-id my-gpu-01
./CortexNode jobs       --node-id my-gpu-01
./CortexNode balance    --node-id my-gpu-01
./CortexNode assignment --node-id my-gpu-01
./CortexNode leave      --node-id my-gpu-01
./CortexNode join       --node-id my-gpu-01
```

The signed assignment records the model version, stage, layer range, memory
requirement, and weights digest. On a selected route, llama.cpp streams the
assigned tensors into the worker's RAM/VRAM and reuses its cache. Adding memory
allows larger stage sets; adding replicas and compute increases throughput.

## CTX and network status

CTX balances live on-chain at each provider's reward address. The controlling
key is local: `.provider/NODE_ID/node_wallet.json` for dedicated workers and
`.provider/cortexterm/node_wallet.json` for the default CortexGrid Chat
identity. The old directory name is retained for wallet compatibility. Back it
up securely.

```bash
curl http://COORDINATOR_IP:65534/api/network
curl http://COORDINATOR_IP:65534/api/models
curl http://COORDINATOR_IP:65534/api/chain
curl http://COORDINATOR_IP:65534/api/economics
```

Every selected provider earns `0.0001 CTX × verified work units`. More
successful processing earns more CTX; idle or invalid nodes earn nothing.
End-user billing is separate from the token ledger. There are no CTX inference
fees, fee splits, burns, epochs, halvings, or block jackpots. See
[docs/ECONOMICS.md](docs/ECONOMICS.md).

## Verify and current boundary

```bash
make test
make test-real
```

The included real-model demo uses SmolLM2 135M; the architecture is the path to
much larger manifests, not a claim that the devnet is production-ready.
Registration, assignments, participation access, work tickets, and receipts
are signed. Upstream llama.cpp RPC tensor transport is still unauthenticated,
so provider ports must remain on a trusted private network or VPN. Worker
receipts prove attributable participation, not Byzantine-safe numerical
correctness. CTX has no guaranteed market value or exchange listing.
