from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Iterable

from .executor import ExecutionReceipt, verify_receipt_signature
from .util import canonical_json, merkle_root, sha256, utc_now
from .wallet import address_from_public_key, transfer_message, verify_signature


TOKEN_UNIT = 1_000_000
TOKEN_SYMBOL = "CTX"
MAX_SUPPLY_MICRO = 1_000_000_000 * TOKEN_UNIT
GENESIS_SUPPLY_MICRO = 1_000_000 * TOKEN_UNIT
DEMO_BALANCE_MICRO = 10_000 * TOKEN_UNIT
INFERENCE_FEE_MICRO = 1 * TOKEN_UNIT
INFERENCE_EMISSION_MICRO = 200_000
PROTOCOL_ADDRESS = "ctx_protocol_treasury"
ESCROW_ADDRESS = "ctx_inference_escrow"


def tokens(amount_micro: int) -> float:
    return round(amount_micro / TOKEN_UNIT, 6)


def receipt_digest(receipt: ExecutionReceipt) -> str:
    return sha256(f"{receipt.canonical_message()}|{receipt.signature}")


def verify_proof_bundle(proof: dict) -> dict:
    if proof.get("algorithm") != "ed25519-receipt-merkle-v1":
        return {"valid": False, "reason": "unsupported proof algorithm"}
    try:
        receipts = [
            ExecutionReceipt.from_dict(receipt) for receipt in proof["receipts"]
        ]
    except (KeyError, TypeError, ValueError):
        return {"valid": False, "reason": "malformed receipt bundle"}
    if not receipts:
        return {"valid": False, "reason": "proof contains no receipts"}

    signature_results = [verify_receipt_signature(receipt) for receipt in receipts]
    computed_root = merkle_root([receipt_digest(receipt) for receipt in receipts])
    chain_valid = all(
        right.input_digest == left.output_digest
        for left, right in zip(receipts, receipts[1:])
    )
    stage_valid = all(
        receipt.shard_index == index for index, receipt in enumerate(receipts)
    )
    root_valid = computed_root == proof.get("root")
    return {
        "valid": all(signature_results) and root_valid and chain_valid and stage_valid,
        "rootValid": root_valid,
        "signaturesValid": all(signature_results),
        "activationChainValid": chain_valid,
        "stageOrderValid": stage_valid,
        "receiptCount": len(receipts),
        "computedRoot": computed_root,
    }


def inference_transactions(
    *,
    proof: dict,
    payer: str,
    inference_id: str,
    current_supply_micro: int,
) -> list[dict]:
    receipts = [ExecutionReceipt.from_dict(receipt) for receipt in proof["receipts"]]
    work_by_address: dict[str, dict] = {}
    for receipt in receipts:
        address = address_from_public_key(receipt.public_key)
        entry = work_by_address.setdefault(
            address, {"workUnits": 0, "nodeIds": set()}
        )
        entry["workUnits"] += receipt.compute_units
        entry["nodeIds"].add(receipt.node_id)

    burn_amount = INFERENCE_FEE_MICRO * 8 // 100
    protocol_amount = INFERENCE_FEE_MICRO * 12 // 100
    provider_fee_pool = INFERENCE_FEE_MICRO - burn_amount - protocol_amount
    remaining_supply = MAX_SUPPLY_MICRO - current_supply_micro
    issuance_pool = min(INFERENCE_EMISSION_MICRO, max(0, remaining_supply))
    total_work = sum(entry["workUnits"] for entry in work_by_address.values())
    if total_work <= 0:
        raise ValueError("inference proof contains no useful work")

    transactions = [
        {
            "type": "inference_fee",
            "from": payer,
            "to": ESCROW_ADDRESS,
            "amountMicro": INFERENCE_FEE_MICRO,
            "inferenceId": inference_id,
        },
        {
            "type": "burn",
            "from": ESCROW_ADDRESS,
            "amountMicro": burn_amount,
            "inferenceId": inference_id,
        },
        {
            "type": "protocol_fee",
            "from": ESCROW_ADDRESS,
            "to": PROTOCOL_ADDRESS,
            "amountMicro": protocol_amount,
            "inferenceId": inference_id,
        },
    ]
    distributed_fee = 0
    distributed_issuance = 0
    ordered_providers = sorted(work_by_address.items())
    for index, (address, work) in enumerate(ordered_providers):
        is_last = index == len(ordered_providers) - 1
        fee_reward = (
            provider_fee_pool - distributed_fee
            if is_last
            else provider_fee_pool * work["workUnits"] // total_work
        )
        issuance_reward = (
            issuance_pool - distributed_issuance
            if is_last
            else issuance_pool * work["workUnits"] // total_work
        )
        distributed_fee += fee_reward
        distributed_issuance += issuance_reward
        transactions.append(
            {
                "type": "provider_reward",
                "to": address,
                "nodeIds": sorted(work["nodeIds"]),
                "workUnits": work["workUnits"],
                "feeAmountMicro": fee_reward,
                "issuanceAmountMicro": issuance_reward,
                "inferenceId": inference_id,
            }
        )
    return transactions


class Blockchain:
    """Small append-only chain whose only issuance path is useful inference."""

    def __init__(self, chain_path: Path, demo_address: str, model_manifest: dict):
        self.chain_path = chain_path
        self.demo_address = demo_address
        self.model_manifest = {
            "id": model_manifest["id"],
            "version": model_manifest["version"],
            "shardCount": model_manifest["shardCount"],
            "shards": [
                {
                    "index": shard["index"],
                    "weightsDigest": shard["weightsDigest"],
                    "computeUnits": shard["computeUnits"],
                }
                for shard in model_manifest["shards"]
            ],
        }
        self.blocks: list[dict] = []
        self.balances: dict[str, int] = {}
        self.nonces: dict[str, int] = {}
        self.total_supply_micro = 0
        self.total_burned_micro = 0
        self.total_provider_rewards_micro = 0
        self._lock = threading.RLock()

        self.chain_path.parent.mkdir(parents=True, exist_ok=True)
        if self.chain_path.exists() and self.chain_path.stat().st_size:
            self._load()
        else:
            self._create_genesis()

    def _create_genesis(self) -> None:
        block = {
            "version": 1,
            "height": 0,
            "previousHash": "0" * 64,
            "timestamp": utc_now(),
            "kind": "genesis",
            "producer": "cortex-grid-devnet",
            "payload": {
                "network": "CortexGrid Devnet",
                "consensus": "proof-of-useful-inference",
                "token": {
                    "name": "Cortex Token",
                    "symbol": TOKEN_SYMBOL,
                    "decimals": 6,
                    "maxSupplyMicro": MAX_SUPPLY_MICRO,
                },
                "models": [self.model_manifest],
            },
            "transactions": [
                {
                    "type": "genesis",
                    "to": PROTOCOL_ADDRESS,
                    "amountMicro": GENESIS_SUPPLY_MICRO - DEMO_BALANCE_MICRO,
                },
                {
                    "type": "genesis",
                    "to": self.demo_address,
                    "amountMicro": DEMO_BALANCE_MICRO,
                },
            ],
        }
        self._finalize_and_append(block)

    def _load(self) -> None:
        lines = self.chain_path.read_text(encoding="utf-8").splitlines()
        self.blocks = [json.loads(line) for line in lines if line.strip()]
        if not self.blocks:
            self._create_genesis()
            return
        genesis_allocations = self.blocks[0].get("transactions", [])
        stored_demo_addresses = {
            tx.get("to")
            for tx in genesis_allocations
            if tx.get("type") == "genesis" and tx.get("to") != PROTOCOL_ADDRESS
        }
        if self.demo_address not in stored_demo_addresses:
            raise RuntimeError("Stored chain does not match the configured demo wallet")
        stored_models = self.blocks[0].get("payload", {}).get("models")
        if canonical_json(stored_models) != canonical_json([self.model_manifest]):
            raise RuntimeError("Stored chain does not match the configured model manifest")
        validation = self.validate()
        if not validation["valid"]:
            raise RuntimeError(f"Stored chain is invalid: {validation['reason']}")
        self._rebuild_state()

    @staticmethod
    def _block_hash(block: dict) -> str:
        content = {key: value for key, value in block.items() if key != "hash"}
        return sha256(canonical_json(content))

    def _finalize_and_append(self, block: dict) -> dict:
        block["hash"] = self._block_hash(block)
        self._apply_transactions(block["transactions"])
        self.blocks.append(block)
        with self.chain_path.open("a", encoding="utf-8") as chain_file:
            chain_file.write(canonical_json(block) + "\n")
            chain_file.flush()
            os.fsync(chain_file.fileno())
        return block

    def _apply_transactions(
        self,
        transactions: Iterable[dict],
        *,
        balances: dict[str, int] | None = None,
        nonces: dict[str, int] | None = None,
        counters: dict[str, int] | None = None,
    ) -> None:
        balances = self.balances if balances is None else balances
        nonces = self.nonces if nonces is None else nonces
        if counters is None:
            counters = {
                "supply": self.total_supply_micro,
                "burned": self.total_burned_micro,
                "rewards": self.total_provider_rewards_micro,
            }

        def debit(address: str, amount: int) -> None:
            if amount < 0 or balances.get(address, 0) < amount:
                raise ValueError(f"insufficient CTX balance for {address}")
            balances[address] = balances.get(address, 0) - amount

        for transaction in transactions:
            kind = transaction["type"]
            amount = int(transaction.get("amountMicro", 0))
            if kind == "genesis":
                balances[transaction["to"]] = balances.get(transaction["to"], 0) + amount
                counters["supply"] += amount
            elif kind in {"transfer", "inference_fee", "protocol_fee"}:
                source = transaction["from"]
                target = transaction["to"]
                if kind == "transfer":
                    expected_nonce = nonces.get(source, 0)
                    if int(transaction["nonce"]) != expected_nonce:
                        raise ValueError("transfer nonce is invalid")
                    if address_from_public_key(transaction["publicKey"]) != source:
                        raise ValueError("transfer public key does not own source address")
                    message = transfer_message(
                        source, target, amount, int(transaction["nonce"])
                    )
                    if not verify_signature(
                        transaction["publicKey"], message, transaction["signature"]
                    ):
                        raise ValueError("transfer signature is invalid")
                    nonces[source] = expected_nonce + 1
                debit(source, amount)
                balances[target] = balances.get(target, 0) + amount
            elif kind == "burn":
                debit(transaction["from"], amount)
                counters["supply"] -= amount
                counters["burned"] += amount
            elif kind == "provider_reward":
                fee_amount = int(transaction["feeAmountMicro"])
                issuance_amount = int(transaction["issuanceAmountMicro"])
                debit(ESCROW_ADDRESS, fee_amount)
                reward = fee_amount + issuance_amount
                balances[transaction["to"]] = balances.get(transaction["to"], 0) + reward
                counters["supply"] += issuance_amount
                counters["rewards"] += reward
            else:
                raise ValueError(f"unknown transaction type: {kind}")

        if counters["supply"] > MAX_SUPPLY_MICRO:
            raise ValueError("CTX maximum supply exceeded")
        if balances.get(ESCROW_ADDRESS, 0) < 0:
            raise ValueError("inference escrow cannot be negative")
        if balances is self.balances:
            self.total_supply_micro = counters["supply"]
            self.total_burned_micro = counters["burned"]
            self.total_provider_rewards_micro = counters["rewards"]

    def _rebuild_state(self) -> None:
        self.balances = {}
        self.nonces = {}
        self.total_supply_micro = 0
        self.total_burned_micro = 0
        self.total_provider_rewards_micro = 0
        for block in self.blocks:
            self._apply_transactions(block["transactions"])

    def commit_inference(
        self,
        *,
        inference_id: str,
        payer: str,
        model: dict,
        proof: dict,
        input_digest: str,
        output_digest: str,
    ) -> dict:
        with self._lock:
            proof_check = verify_proof_bundle(proof)
            if not proof_check["valid"]:
                raise ValueError("cannot mint CTX from an invalid inference proof")
            if self.balances.get(payer, 0) < INFERENCE_FEE_MICRO:
                raise ValueError(f"insufficient {TOKEN_SYMBOL} for inference")

            receipts = [
                ExecutionReceipt.from_dict(receipt) for receipt in proof["receipts"]
            ]
            transactions = inference_transactions(
                proof=proof,
                payer=payer,
                inference_id=inference_id,
                current_supply_micro=self.total_supply_micro,
            )

            block = {
                "version": 1,
                "height": len(self.blocks),
                "previousHash": self.blocks[-1]["hash"],
                "timestamp": utc_now(),
                "kind": "inference",
                "producer": "proof-of-useful-inference",
                "payload": {
                    "inferenceId": inference_id,
                    "model": {
                        "id": model["id"],
                        "version": model["version"],
                        "shardCount": model["shardCount"],
                    },
                    "inputDigest": input_digest,
                    "outputDigest": output_digest,
                    "usefulWorkUnits": sum(
                        receipt.compute_units for receipt in receipts
                    ),
                    "proof": proof,
                },
                "transactions": transactions,
            }
            return copy.deepcopy(self._finalize_and_append(block))

    def commit_transfer(self, transaction: dict) -> dict:
        with self._lock:
            normalized = {
                "type": "transfer",
                "from": transaction["from"],
                "to": transaction["to"],
                "amountMicro": int(transaction["amountMicro"]),
                "nonce": int(transaction["nonce"]),
                "publicKey": transaction["publicKey"],
                "signature": transaction["signature"],
            }
            block = {
                "version": 1,
                "height": len(self.blocks),
                "previousHash": self.blocks[-1]["hash"],
                "timestamp": utc_now(),
                "kind": "transfer",
                "producer": normalized["from"],
                "payload": {},
                "transactions": [normalized],
            }
            return copy.deepcopy(self._finalize_and_append(block))

    def validate(self) -> dict:
        balances: dict[str, int] = {}
        nonces: dict[str, int] = {}
        counters = {"supply": 0, "burned": 0, "rewards": 0}
        expected_previous = "0" * 64
        try:
            for height, block in enumerate(self.blocks):
                if block["height"] != height:
                    raise ValueError(f"invalid height at block {height}")
                if block["previousHash"] != expected_previous:
                    raise ValueError(f"invalid previous hash at block {height}")
                if self._block_hash(block) != block["hash"]:
                    raise ValueError(f"invalid hash at block {height}")
                if block["kind"] == "genesis":
                    if height != 0:
                        raise ValueError("genesis block must be first")
                    genesis_total = sum(
                        int(transaction["amountMicro"])
                        for transaction in block["transactions"]
                        if transaction["type"] == "genesis"
                    )
                    if (
                        len(block["transactions"]) != 2
                        or genesis_total != GENESIS_SUPPLY_MICRO
                    ):
                        raise ValueError("genesis allocation is invalid")
                    expected_genesis = [
                        {
                            "type": "genesis",
                            "to": PROTOCOL_ADDRESS,
                            "amountMicro": GENESIS_SUPPLY_MICRO
                            - DEMO_BALANCE_MICRO,
                        },
                        {
                            "type": "genesis",
                            "to": self.demo_address,
                            "amountMicro": DEMO_BALANCE_MICRO,
                        },
                    ]
                    if canonical_json(block["transactions"]) != canonical_json(
                        expected_genesis
                    ):
                        raise ValueError("genesis recipients are invalid")
                    if canonical_json(
                        block["payload"].get("models")
                    ) != canonical_json([self.model_manifest]):
                        raise ValueError("genesis model registry is invalid")
                elif block["kind"] == "inference":
                    proof_check = verify_proof_bundle(block["payload"]["proof"])
                    if not proof_check["valid"]:
                        raise ValueError(f"invalid inference proof at block {height}")
                    receipts = [
                        ExecutionReceipt.from_dict(receipt)
                        for receipt in block["payload"]["proof"]["receipts"]
                    ]
                    payload = block["payload"]
                    inference_id = payload["inferenceId"]
                    model = payload["model"]
                    if not all(
                        receipt.request_id == inference_id
                        and receipt.model_id == model["id"]
                        and receipt.model_version == model["version"]
                        for receipt in receipts
                    ):
                        raise ValueError(f"receipt identity mismatch at block {height}")
                    if (
                        len(receipts) != int(model["shardCount"])
                        or payload["inputDigest"] != receipts[0].input_digest
                        or payload["outputDigest"] != receipts[-1].output_digest
                        or int(payload["usefulWorkUnits"])
                        != sum(receipt.compute_units for receipt in receipts)
                    ):
                        raise ValueError(f"inference payload mismatch at block {height}")
                    if (
                        model["id"] != self.model_manifest["id"]
                        or model["version"] != self.model_manifest["version"]
                        or int(model["shardCount"])
                        != self.model_manifest["shardCount"]
                    ):
                        raise ValueError(f"model is not registered at block {height}")
                    for receipt, registered_shard in zip(
                        receipts, self.model_manifest["shards"]
                    ):
                        if (
                            receipt.shard_index != registered_shard["index"]
                            or receipt.weights_digest
                            != registered_shard["weightsDigest"]
                            or receipt.compute_units
                            != registered_shard["computeUnits"]
                        ):
                            raise ValueError(
                                f"model work commitment mismatch at block {height}"
                            )
                    if (
                        not block["transactions"]
                        or block["transactions"][0]["type"] != "inference_fee"
                    ):
                        raise ValueError(f"inference fee missing at block {height}")
                    expected = inference_transactions(
                        proof=payload["proof"],
                        payer=block["transactions"][0]["from"],
                        inference_id=inference_id,
                        current_supply_micro=counters["supply"],
                    )
                    if canonical_json(expected) != canonical_json(
                        block["transactions"]
                    ):
                        raise ValueError(
                            f"inference settlement is invalid at block {height}"
                        )
                elif block["kind"] == "transfer":
                    if (
                        len(block["transactions"]) != 1
                        or block["transactions"][0]["type"] != "transfer"
                    ):
                        raise ValueError(f"invalid transfer block at height {height}")
                else:
                    raise ValueError(f"unknown block kind at height {height}")
                self._apply_transactions(
                    block["transactions"],
                    balances=balances,
                    nonces=nonces,
                    counters=counters,
                )
                if sum(balances.values()) != counters["supply"]:
                    raise ValueError(f"token conservation failed at block {height}")
                expected_previous = block["hash"]
        except (KeyError, TypeError, ValueError) as error:
            return {"valid": False, "height": len(self.blocks) - 1, "reason": str(error)}
        return {
            "valid": True,
            "height": len(self.blocks) - 1,
            "latestHash": expected_previous,
            "reason": None,
        }

    def balance_of(self, address: str) -> int:
        return self.balances.get(address, 0)

    def nonce_of(self, address: str) -> int:
        return self.nonces.get(address, 0)

    @staticmethod
    def public_block(block: dict, *, include_receipts: bool = False) -> dict:
        result = copy.deepcopy(block)
        if result["kind"] == "inference" and not include_receipts:
            proof = result["payload"]["proof"]
            result["payload"]["proof"] = {
                "algorithm": proof["algorithm"],
                "root": proof["root"],
                "receiptCount": len(proof["receipts"]),
            }
        for transaction in result["transactions"]:
            for key in ("amountMicro", "feeAmountMicro", "issuanceAmountMicro"):
                if key in transaction:
                    transaction[key.removesuffix("Micro")] = tokens(transaction[key])
        return result

    def snapshot(self) -> dict:
        with self._lock:
            balances = [
                {"address": address, "balance": tokens(amount)}
                for address, amount in self.balances.items()
                if amount > 0 and address != ESCROW_ADDRESS
            ]
            balances.sort(key=lambda item: item["balance"], reverse=True)
            return {
                "network": "CortexGrid Devnet",
                "consensus": "proof-of-useful-inference",
                "height": len(self.blocks) - 1,
                "latestBlockHash": self.blocks[-1]["hash"],
                "token": {
                    "name": "Cortex Token",
                    "symbol": TOKEN_SYMBOL,
                    "decimals": 6,
                    "maxSupply": tokens(MAX_SUPPLY_MICRO),
                    "totalSupply": tokens(self.total_supply_micro),
                    "totalBurned": tokens(self.total_burned_micro),
                    "providerRewards": tokens(self.total_provider_rewards_micro),
                },
                "demoWallet": {
                    "address": self.demo_address,
                    "balance": tokens(self.balance_of(self.demo_address)),
                    "nonce": self.nonce_of(self.demo_address),
                },
                "balances": balances,
                "recentBlocks": [
                    self.public_block(block)
                    for block in reversed(self.blocks[-12:])
                ],
                "valid": self.validate()["valid"],
            }
