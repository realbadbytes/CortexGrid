from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .blockchain import (
    Blockchain,
    INFERENCE_FEE_MICRO,
    TOKEN_SYMBOL,
    receipt_digest,
    tokens,
    verify_proof_bundle,
)
from .executor import ExecutionReceipt, NativeWorker, verify_receipt_signature
from .model_runtime import DistributedModelRuntime
from .util import merkle_root, sha256, sha256_file, utc_now
from .wallet import Wallet


NODE_BLUEPRINTS = [
    ("atlas", "Atlas", "Northstar Labs", "us-east", 24, 18, "#8da2fb"),
    ("ember", "Ember", "Independent", "us-west", 24, 31, "#ff8b5b"),
    ("kite", "Kite", "Tensor House", "eu-west", 24, 44, "#5ee3b1"),
    ("orbit", "Orbit", "Lattice DAO", "eu-central", 24, 51, "#d99cff"),
    ("sable", "Sable", "Independent", "ap-south", 24, 72, "#ffd166"),
    ("nova", "Nova", "Compute Union", "ap-northeast", 24, 81, "#60c7ff"),
]


def _synthetic_model_manifest() -> dict:
    model = {
        "id": "cortex-72b-devnet",
        "name": "Cortex 72B · Devnet",
        "version": "0.1.0",
        "parameters": 72_000_000_000,
        "precision": "4-bit",
        "architecture": "pipeline-parallel",
        "shardCount": 12,
        "shardSizeGiB": 3.5,
        "replicationFactor": 2,
        "contextWindow": 32_768,
        "status": "ready",
    }
    model["shards"] = [
        {
            "index": index,
            "sizeGiB": model["shardSizeGiB"],
            "computeUnits": 2 + index % 3,
            "weightsDigest": sha256(
                f"cortex-grid-weights|{model['id']}|{model['version']}|{index}"
            ),
        }
        for index in range(model["shardCount"])
    ]
    return model


def _real_model_manifest(model_path: Path) -> dict:
    model_digest = sha256_file(model_path)
    model_size_gib = model_path.stat().st_size / (1024**3)
    model = {
        "id": "smollm2-135m-rpc",
        "name": "SmolLM2 135M · Distributed",
        "version": f"q4km-{model_digest[:12]}",
        "parameters": 135_000_000,
        "precision": "Q4_K_M",
        "architecture": "llama.cpp-rpc-layer",
        "shardCount": 2,
        "shardSizeGiB": round(model_size_gib / 2, 6),
        "replicationFactor": 3,
        "contextWindow": 1_024,
        "status": "ready",
        "modelDigest": model_digest,
        "executionMode": "real-transformer",
    }
    model["shards"] = [
        {
            "index": index,
            "sizeGiB": model["shardSizeGiB"],
            "computeUnits": 15,
            "layerRange": [index * 15, index * 15 + 14],
            "weightsDigest": sha256(
                f"{model_digest}|pipeline-stage|{index}|layers-{index * 15}-"
                f"{index * 15 + 14}"
            ),
        }
        for index in range(model["shardCount"])
    ]
    return model


@dataclass
class ComputeNode:
    id: str
    name: str
    operator: str
    region: str
    memory_gib: float
    network_latency_ms: int
    accent: str
    worker: NativeWorker
    status: str = "online"
    completed_jobs: int = 0
    failed_proofs: int = 0
    shards: list[dict] = field(default_factory=list)

    @property
    def allocated_memory_gib(self) -> float:
        return sum(shard["sizeGiB"] for shard in self.shards)

    def hosts(self, model_id: str, shard_index: int) -> bool:
        return any(
            shard["modelId"] == model_id and shard["index"] == shard_index
            for shard in self.shards
        )

    def snapshot(self, chain: Blockchain) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "operator": self.operator,
            "region": self.region,
            "status": self.status,
            "memoryGiB": self.memory_gib,
            "allocatedMemoryGiB": self.allocated_memory_gib,
            "freeMemoryGiB": self.memory_gib - self.allocated_memory_gib,
            "networkLatencyMs": self.network_latency_ms,
            "completedJobs": self.completed_jobs,
            "failedProofs": self.failed_proofs,
            "backend": self.worker.capabilities,
            "publicKey": self.worker.public_key,
            "rewardAddress": self.worker.reward_address,
            "balance": tokens(chain.balance_of(self.worker.reward_address)),
            "accent": self.accent,
            "shards": [
                {
                    "modelId": shard["modelId"],
                    "index": shard["index"],
                    "sizeGiB": shard["sizeGiB"],
                    "weightsDigest": shard["weightsDigest"],
                }
                for shard in self.shards
            ],
        }


class CortexNetwork:
    """The single Python control plane: route, verify, finalize a block."""

    def __init__(
        self,
        *,
        executor_path: Path | None = None,
        data_dir: Path | None = None,
        real_model: bool | None = None,
    ):
        project_root = Path(__file__).resolve().parents[2]
        self.executor_path = executor_path or Path(
            os.environ.get(
                "CORTEX_EXECUTOR_PATH", project_root / "build" / "cortex_worker"
            )
        )
        self.data_dir = data_dir or Path(
            os.environ.get("CORTEX_DATA_DIR", project_root / ".data")
        )
        self.model_path = Path(
            os.environ.get(
                "CORTEX_MODEL_PATH",
                project_root / "models" / "SmolLM2-135M-Instruct-Q4_K_M.gguf",
            )
        )
        self.rpc_binary = Path(
            os.environ.get(
                "CORTEX_RPC_BINARY",
                project_root / "build" / "llama" / "bin" / "ggml-rpc-server",
            )
        )
        self.model_server_binary = Path(
            os.environ.get(
                "CORTEX_MODEL_SERVER_BINARY",
                project_root / "build" / "llama" / "bin" / "llama-server",
            )
        )
        runtime_available = DistributedModelRuntime.is_available(
            self.model_path, self.rpc_binary, self.model_server_binary
        )
        if real_model is None:
            runtime_setting = os.environ.get("CORTEX_REAL_MODEL", "auto").lower()
            real_model = runtime_available if runtime_setting == "auto" else (
                runtime_setting in {"1", "true", "on", "yes"}
            )
        if real_model and not runtime_available:
            raise RuntimeError(
                "real model mode requested, but the GGUF or llama.cpp binaries "
                "are missing; run `make bootstrap-model`"
            )

        self.real_model_enabled = bool(real_model)
        self.model = (
            _real_model_manifest(self.model_path)
            if self.real_model_enabled
            else _synthetic_model_manifest()
        )
        self.demo_wallet = Wallet.load_or_create(self.data_dir / "demo_wallet.json")
        chain_filename = (
            f"chain-{self.model['id']}.jsonl"
            if self.real_model_enabled
            else "chain.jsonl"
        )
        self.chain = Blockchain(
            self.data_dir / chain_filename,
            demo_address=self.demo_wallet.address,
            model_manifest=self.model,
        )
        self.nodes: dict[str, ComputeNode] = {}
        self.model_runtime: DistributedModelRuntime | None = None
        self.events: list[dict] = []
        self.recent_inferences: list[dict] = []
        self._lock = threading.RLock()
        self._closed = False

        try:
            self._start_nodes()
            self._place_shards()
            if self.real_model_enabled:
                self.model_runtime = DistributedModelRuntime(
                    model_path=self.model_path,
                    rpc_binary=self.rpc_binary,
                    server_binary=self.model_server_binary,
                    node_ids=list(self.nodes),
                    data_dir=self.data_dir / "runtime",
                )
        except Exception:
            self.close()
            raise
        self._event(
            "network",
            "Execution network online",
            f"{len(self.nodes)} native workers · {self.model['shardCount']} shards · "
            f"{self.model['replicationFactor']}× replicas",
        )

    def _start_nodes(self) -> None:
        for blueprint in NODE_BLUEPRINTS:
            node_id, name, operator, region, memory, latency, accent = blueprint
            worker = NativeWorker(node_id, self.executor_path)
            self.nodes[node_id] = ComputeNode(
                id=node_id,
                name=name,
                operator=operator,
                region=region,
                memory_gib=memory,
                network_latency_ms=latency,
                accent=accent,
                worker=worker,
            )

    def _place_shards(self) -> None:
        nodes = list(self.nodes.values())
        for shard in self.model["shards"]:
            stride = max(1, len(nodes) // self.model["replicationFactor"])
            replica_indices = [
                (shard["index"] + replica * stride) % len(nodes)
                for replica in range(self.model["replicationFactor"])
            ]
            for replica_index in replica_indices:
                nodes[replica_index].shards.append(
                    {"modelId": self.model["id"], **shard}
                )

    def _event(self, kind: str, title: str, detail: str, **metadata: object) -> None:
        self.events.insert(
            0,
            {
                "id": f"evt_{secrets.token_hex(6)}",
                "type": kind,
                "title": title,
                "detail": detail,
                "timestamp": utc_now(),
                **metadata,
            },
        )
        del self.events[100:]

    def _route(self, shard_index: int) -> ComputeNode:
        candidates = [
            node
            for node in self.nodes.values()
            if node.status == "online" and node.hosts(self.model["id"], shard_index)
        ]
        if not candidates:
            raise RuntimeError(f"no online replica for shard {shard_index}")
        return min(
            candidates,
            key=lambda node: (
                node.network_latency_ms,
                node.completed_jobs,
                node.id,
            ),
        )

    def _verify_current_receipt(
        self,
        receipt: ExecutionReceipt,
        node: ComputeNode,
        shard: dict,
        expected_input: str,
    ) -> None:
        checks = {
            "node identity": (
                receipt.node_id == node.id
                and receipt.public_key == node.worker.public_key
            ),
            "model identity": (
                receipt.model_id == self.model["id"]
                and receipt.model_version == self.model["version"]
            ),
            "shard assignment": (
                receipt.shard_index == shard["index"]
                and node.hosts(self.model["id"], shard["index"])
            ),
            "weight commitment": receipt.weights_digest == shard["weightsDigest"],
            "activation chain": receipt.input_digest == expected_input,
            "worker signature": verify_receipt_signature(receipt),
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            node.failed_proofs += 1
            raise RuntimeError(f"rejected shard receipt: {', '.join(failed)}")

    @staticmethod
    def _completion(prompt: str, proof_root: str) -> str:
        normalized = prompt.lower()
        if "token" in normalized or "econom" in normalized:
            body = (
                "CTX is consumed for inference and paid to the workers that keep model "
                "shards resident and execute the route. Useful-work issuance is capped; "
                "transfers never create supply. Utility can create demand, but market "
                "price remains an external outcome."
            )
        elif any(word in normalized for word in ("scale", "ram", "parameter")):
            body = (
                "A model larger than one host is divided by transformer stage. Each "
                "activation moves to the next selected replica, while signed receipts "
                "bind the model version, weight commitment, input, output, and work."
            )
        else:
            body = (
                "The native execution route completed and every stage signature was "
                "verified before the control plane finalized its useful-work block. "
                "This devnet uses deterministic activation kernels; real tensor weights "
                "can replace that kernel without changing the chain contract."
            )
        return f"{body}\n\nBlock proof: {proof_root[:16]}…"

    def _execute_synthetic_route(
        self, inference_id: str, input_digest: str
    ) -> tuple[list[ExecutionReceipt], list[dict], str, dict]:
        activation_digest = input_digest
        receipts: list[ExecutionReceipt] = []
        route: list[dict] = []
        for shard in self.model["shards"]:
            node = self._route(shard["index"])
            stage_started = time.perf_counter()
            receipt = node.worker.execute(
                request_id=inference_id,
                model_id=self.model["id"],
                model_version=self.model["version"],
                shard_index=shard["index"],
                input_digest=activation_digest,
                weights_digest=shard["weightsDigest"],
                compute_units=shard["computeUnits"],
                challenge=secrets.token_hex(16),
            )
            self._verify_current_receipt(receipt, node, shard, activation_digest)
            stage_elapsed_ms = round((time.perf_counter() - stage_started) * 1000, 2)
            node.completed_jobs += 1
            receipts.append(receipt)
            activation_digest = receipt.output_digest
            route.append(
                {
                    "stage": shard["index"],
                    "nodeId": node.id,
                    "nodeName": node.name,
                    "region": node.region,
                    "backend": receipt.backend,
                    "kernelMicros": receipt.kernel_micros,
                    "controlPlaneMs": stage_elapsed_ms,
                    "networkLatencyMs": node.network_latency_ms,
                    "computeUnits": receipt.compute_units,
                    "receiptDigest": receipt_digest(receipt),
                }
            )
        return receipts, route, activation_digest, {
            "mode": "deterministic-kernel",
            "proofLevel": "signed-execution-receipts",
        }

    def _execute_real_route(
        self, inference_id: str, prompt: str, input_digest: str
    ) -> tuple[list[ExecutionReceipt], list[dict], str, dict]:
        if self.model_runtime is None:
            raise RuntimeError("real model runtime is unavailable")
        selected_nodes = [
            self._route(shard["index"]) for shard in self.model["shards"]
        ]
        if len({node.id for node in selected_nodes}) != len(selected_nodes):
            raise RuntimeError("distributed route requires distinct execution nodes")

        started = time.perf_counter()
        generation = self.model_runtime.generate(
            prompt=prompt,
            route=[node.id for node in selected_nodes],
        )
        runtime_elapsed_ms = (time.perf_counter() - started) * 1000
        completion_digest = sha256(generation["text"])
        activation_digest = input_digest
        receipts: list[ExecutionReceipt] = []
        route: list[dict] = []
        total_kernel_micros = int(
            (
                generation["timings"].get("prompt_ms", 0)
                + generation["timings"].get("predicted_ms", 0)
            )
            * 1_000
        )
        per_stage_micros = max(
            1, total_kernel_micros // len(self.model["shards"])
        )

        for index, (shard, node) in enumerate(
            zip(self.model["shards"], selected_nodes)
        ):
            is_last = index == len(self.model["shards"]) - 1
            output_digest = (
                completion_digest
                if is_last
                else sha256(
                    f"{inference_id}|{activation_digest}|{completion_digest}|"
                    f"rpc-boundary|{index}"
                )
            )
            receipt = node.worker.attest(
                request_id=inference_id,
                model_id=self.model["id"],
                model_version=self.model["version"],
                shard_index=shard["index"],
                input_digest=activation_digest,
                output_digest=output_digest,
                weights_digest=shard["weightsDigest"],
                compute_units=shard["computeUnits"],
                challenge=secrets.token_hex(16),
                backend="llama.cpp-rpc",
                kernel_micros=per_stage_micros,
            )
            self._verify_current_receipt(receipt, node, shard, activation_digest)
            node.completed_jobs += 1
            receipts.append(receipt)
            activation_digest = output_digest
            route.append(
                {
                    "stage": shard["index"],
                    "layers": shard["layerRange"],
                    "nodeId": node.id,
                    "nodeName": node.name,
                    "region": node.region,
                    "backend": receipt.backend,
                    "kernelMicros": receipt.kernel_micros,
                    "controlPlaneMs": round(runtime_elapsed_ms / len(selected_nodes), 2),
                    "networkLatencyMs": node.network_latency_ms,
                    "computeUnits": receipt.compute_units,
                    "receiptDigest": receipt_digest(receipt),
                }
            )

        return receipts, route, activation_digest, {
            "mode": "real-transformer",
            "proofLevel": "signed-participation-attestation",
            "text": generation["text"],
            "usage": generation["usage"],
            "timings": generation["timings"],
            "runtimeModel": generation["runtimeModel"],
            "systemFingerprint": generation["systemFingerprint"],
        }

    def run_inference(self, prompt: str, model_id: str | None = None) -> dict:
        if not isinstance(prompt, str) or len(prompt.strip()) < 2:
            raise ValueError("prompt must contain at least 2 characters")
        if len(prompt) > 8_000:
            raise ValueError("prompt exceeds the 8,000 character devnet limit")
        if model_id not in (None, self.model["id"]):
            raise ValueError(f"unknown model: {model_id}")

        with self._lock:
            inference_id = f"inf_{secrets.token_hex(8)}"
            started = time.perf_counter()
            input_digest = sha256(
                f"{inference_id}|{self.model['id']}|{prompt.strip()}"
            )
            prompt_digest = sha256(prompt.strip())
            if self.model_runtime is None:
                receipts, route, activation_digest, runtime = (
                    self._execute_synthetic_route(inference_id, input_digest)
                )
            else:
                receipts, route, activation_digest, runtime = (
                    self._execute_real_route(
                        inference_id, prompt.strip(), input_digest
                    )
                )

            proof = {
                "algorithm": "ed25519-receipt-merkle-v1",
                "root": merkle_root([receipt_digest(receipt) for receipt in receipts]),
                "receipts": [receipt.to_dict() for receipt in receipts],
            }
            proof_check = verify_proof_bundle(proof)
            if not proof_check["valid"]:
                raise RuntimeError("final proof bundle did not verify")

            block = self.chain.commit_inference(
                inference_id=inference_id,
                payer=self.demo_wallet.address,
                model=self.model,
                proof=proof,
                input_digest=input_digest,
                output_digest=activation_digest,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            reward_transactions = [
                transaction
                for transaction in block["transactions"]
                if transaction["type"] == "provider_reward"
            ]
            accounting = {
                "symbol": TOKEN_SYMBOL,
                "fee": tokens(INFERENCE_FEE_MICRO),
                "burned": sum(
                    tokens(transaction["amountMicro"])
                    for transaction in block["transactions"]
                    if transaction["type"] == "burn"
                ),
                "providerRewards": sum(
                    tokens(
                        transaction["feeAmountMicro"]
                        + transaction["issuanceAmountMicro"]
                    )
                    for transaction in reward_transactions
                ),
                "issuance": sum(
                    tokens(transaction["issuanceAmountMicro"])
                    for transaction in reward_transactions
                ),
            }
            result = {
                "id": inference_id,
                "status": "finalized",
                "simulation": self.model_runtime is None,
                "model": {
                    key: self.model[key]
                    for key in ("id", "name", "version", "shardCount")
                },
                "input": {
                    "prompt": prompt.strip(),
                    "promptDigest": prompt_digest,
                },
                "output": {
                    "text": runtime.get("text")
                    or self._completion(prompt.strip(), proof["root"]),
                    "outputDigest": activation_digest,
                },
                "performance": {
                    "wallTimeMs": elapsed_ms,
                    "stages": len(route),
                    "kernelMicros": sum(item["kernelMicros"] for item in route),
                    "usefulWorkUnits": sum(
                        receipt.compute_units for receipt in receipts
                    ),
                    "tokens": runtime.get("usage", {}),
                    "timings": runtime.get("timings", {}),
                },
                "runtime": {
                    key: value
                    for key, value in runtime.items()
                    if key not in {"text", "usage", "timings"}
                },
                "route": route,
                "proof": proof,
                "block": Blockchain.public_block(block),
                "accounting": accounting,
            }
            self.recent_inferences.insert(
                0,
                {
                    "id": inference_id,
                    "prompt": prompt.strip(),
                    "model": self.model["name"],
                    "blockHeight": block["height"],
                    "blockHash": block["hash"],
                    "proofRoot": proof["root"],
                    "wallTimeMs": elapsed_ms,
                    "completedAt": utc_now(),
                },
            )
            del self.recent_inferences[20:]
            self._event(
                "block",
                f"Block #{block['height']} finalized",
                f"{len(receipts)} receipts · {proof['root'][:12]} · "
                f"{accounting['providerRewards']} {TOKEN_SYMBOL} rewarded",
                inferenceId=inference_id,
                blockHash=block["hash"],
            )
            return result

    def set_node_status(self, node_id: str, status: str) -> dict:
        if status not in {"online", "offline"}:
            raise ValueError("node status must be online or offline")
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise ValueError(f"unknown node: {node_id}")
            node.status = status
            self._event(
                "node" if status == "online" else "warning",
                f"{node.name} is {status}",
                f"{node.region} · {len(node.shards)} resident shards",
                nodeId=node.id,
            )
            return node.snapshot(self.chain)

    def public_model(self) -> dict:
        placements = []
        for shard in self.model["shards"]:
            placements.append(
                {
                    **shard,
                    "replicas": [
                        node.id
                        for node in self.nodes.values()
                        if node.hosts(self.model["id"], shard["index"])
                    ],
                }
            )
        return {
            **{
                key: value
                for key, value in self.model.items()
                if key != "shards"
            },
            "modelMemoryGiB": self.model["shardCount"]
            * self.model["shardSizeGiB"],
            "replicatedMemoryGiB": self.model["shardCount"]
            * self.model["shardSizeGiB"]
            * self.model["replicationFactor"],
            "placements": placements,
        }

    def snapshot(self) -> dict:
        with self._lock:
            node_snapshots = [
                node.snapshot(self.chain) for node in self.nodes.values()
            ]
            online = [node for node in node_snapshots if node["status"] == "online"]
            inference_blocks = [
                block for block in self.chain.blocks if block["kind"] == "inference"
            ]
            return {
                "name": "CortexGrid",
                "environment": "devnet",
                "protocolVersion": "0.1.0",
                "status": (
                    "healthy"
                    if len(online) == len(node_snapshots)
                    else "degraded"
                ),
                "metrics": {
                    "nodesOnline": len(online),
                    "nodesTotal": len(node_snapshots),
                    "totalMemoryGiB": sum(
                        node["memoryGiB"] for node in node_snapshots
                    ),
                    "allocatedMemoryGiB": sum(
                        node["allocatedMemoryGiB"] for node in node_snapshots
                    ),
                    "modelsReady": 1,
                    "inferencesFinalized": len(inference_blocks),
                    "verifiedReceipts": sum(
                        len(block["payload"]["proof"]["receipts"])
                        for block in inference_blocks
                    ),
                    "chainHeight": len(self.chain.blocks) - 1,
                },
                "model": self.public_model(),
                "modelRuntime": (
                    self.model_runtime.snapshot()
                    if self.model_runtime is not None
                    else {
                        "available": False,
                        "active": False,
                        "engine": "cortex-deterministic-kernel",
                    }
                ),
                "nodes": node_snapshots,
                "chain": self.chain.snapshot(),
                "recentEvents": list(self.events[:30]),
                "recentInferences": list(self.recent_inferences),
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.model_runtime is not None:
            self.model_runtime.close()
        for node in self.nodes.values():
            node.worker.close()
