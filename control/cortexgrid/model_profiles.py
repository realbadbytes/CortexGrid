from __future__ import annotations

import json
import os
from pathlib import Path

from .util import sha256


GIB = 1024**3
TIB = 1024**4
FIVE_TIB_BYTES = 5 * TIB
FIVE_TIB_STAGE_COUNT = 160
FIVE_TIB_STAGE_BYTES = FIVE_TIB_BYTES // FIVE_TIB_STAGE_COUNT
FIVE_TIB_STAGE_MEMORY_GIB = 40
FIVE_TIB_LAYER_COUNT = 320


def five_tb_model_manifest() -> dict:
    artifact_base = os.environ.get(
        "CORTEX_5TB_ARTIFACT_BASE",
        "https://models.example.invalid/cortex-5tb",
    ).rstrip("/")
    model_id = "cortex-5tb-theoretical"
    version = "manifest-v1"
    shards = []
    for index in range(FIVE_TIB_STAGE_COUNT):
        first_layer = index * (FIVE_TIB_LAYER_COUNT // FIVE_TIB_STAGE_COUNT)
        last_layer = first_layer + 1
        digest = sha256(
            f"{model_id}|{version}|stage-{index:03d}|"
            f"{FIVE_TIB_STAGE_BYTES}|layers-{first_layer}-{last_layer}"
        )
        shards.append(
            {
                "index": index,
                "sizeBytes": FIVE_TIB_STAGE_BYTES,
                "sizeGiB": FIVE_TIB_STAGE_BYTES / GIB,
                "minimumMemoryGiB": FIVE_TIB_STAGE_MEMORY_GIB,
                "computeUnits": 2,
                "layerRange": [first_layer, last_layer],
                "weightsDigest": digest,
                "artifact": {
                    "url": f"{artifact_base}/{digest}.ctxpack",
                    "sha256": digest,
                    "sizeBytes": FIVE_TIB_STAGE_BYTES,
                    "available": False,
                },
            }
        )
    return {
        "id": model_id,
        "name": "Cortex 5 TiB · Capacity Profile",
        "version": version,
        "parameters": FIVE_TIB_BYTES * 2,
        "precision": "nominal 4-bit",
        "architecture": "pipeline-parallel",
        "layerCount": FIVE_TIB_LAYER_COUNT,
        "shardCount": FIVE_TIB_STAGE_COUNT,
        "shardSizeGiB": FIVE_TIB_STAGE_BYTES / GIB,
        "replicationFactor": 2,
        "contextWindow": 32_768,
        "status": "awaiting-capacity",
        "modelSizeBytes": FIVE_TIB_BYTES,
        "modelSizeTiB": FIVE_TIB_BYTES / TIB,
        "executionMode": "capacity-profile",
        "artifactsAvailable": False,
        "shards": shards,
    }


def validate_model_manifest(manifest: dict) -> dict:
    required = {
        "id",
        "name",
        "version",
        "precision",
        "architecture",
        "shardCount",
        "replicationFactor",
        "contextWindow",
        "shards",
    }
    if not required.issubset(manifest):
        missing = ", ".join(sorted(required - set(manifest)))
        raise ValueError(f"model manifest is missing: {missing}")
    shards = manifest["shards"]
    if (
        not isinstance(shards, list)
        or not shards
        or len(shards) != int(manifest["shardCount"])
    ):
        raise ValueError("model manifest shard count is invalid")
    expected_indices = list(range(len(shards)))
    if [int(shard["index"]) for shard in shards] != expected_indices:
        raise ValueError("model manifest shards must be ordered and contiguous")
    for shard in shards:
        size_bytes = int(
            shard.get(
                "sizeBytes",
                round(float(shard["sizeGiB"]) * GIB),
            )
        )
        minimum_memory = float(
            shard.get("minimumMemoryGiB", size_bytes / GIB)
        )
        if (
            size_bytes <= 0
            or minimum_memory < size_bytes / GIB
            or int(shard["computeUnits"]) <= 0
            or not isinstance(shard["weightsDigest"], str)
            or len(shard["weightsDigest"]) != 64
        ):
            raise ValueError(f"model shard {shard['index']} is invalid")
        shard["sizeBytes"] = size_bytes
        shard["sizeGiB"] = size_bytes / GIB
        shard["minimumMemoryGiB"] = minimum_memory
    manifest["modelSizeBytes"] = sum(
        int(shard["sizeBytes"]) for shard in shards
    )
    manifest["modelSizeTiB"] = manifest["modelSizeBytes"] / TIB
    return manifest


def load_model_manifest(path: Path) -> dict:
    return validate_model_manifest(
        json.loads(path.read_text(encoding="utf-8"))
    )


def capacity_requirements(manifest: dict) -> dict:
    validate_model_manifest(manifest)
    replicas = int(manifest["replicationFactor"])
    minimum_ram_gib = sum(
        float(shard["minimumMemoryGiB"]) for shard in manifest["shards"]
    )
    weight_gib = sum(float(shard["sizeGiB"]) for shard in manifest["shards"])
    largest_stage = max(
        float(shard["minimumMemoryGiB"]) for shard in manifest["shards"]
    )
    return {
        "modelId": manifest["id"],
        "modelSizeTiB": manifest["modelSizeBytes"] / TIB,
        "stageCount": int(manifest["shardCount"]),
        "replicationFactor": replicas,
        "minimumProviderMemoryGiB": largest_stage,
        "minimumProvidersNoReplication": int(manifest["shardCount"]),
        "minimumProvidersAtTargetReplication": (
            int(manifest["shardCount"]) * replicas
        ),
        "weightMemoryGiBNoReplication": weight_gib,
        "weightMemoryGiBAtTargetReplication": weight_gib * replicas,
        "minimumNetworkRamGiBNoReplication": minimum_ram_gib,
        "minimumNetworkRamGiBAtTargetReplication": minimum_ram_gib * replicas,
    }


def capacity_plan(
    manifest: dict, *, provider_count: int, provider_memory_gib: float
) -> dict:
    requirements = capacity_requirements(manifest)
    if provider_count < 0 or provider_memory_gib <= 0:
        raise ValueError("provider capacity values are invalid")
    compatible_stages = sum(
        1
        for shard in manifest["shards"]
        if float(shard["minimumMemoryGiB"]) <= provider_memory_gib
    )
    target_placements = (
        int(manifest["shardCount"]) * int(manifest["replicationFactor"])
    )
    compatible_placements = compatible_stages * int(
        manifest["replicationFactor"]
    )
    return {
        **requirements,
        "providerCount": provider_count,
        "providerMemoryGiB": provider_memory_gib,
        "aggregateProviderRamGiB": provider_count * provider_memory_gib,
        "canHostOneStage": compatible_stages > 0,
        "stagePlacementsPossible": min(
            provider_count, compatible_placements, target_placements
        ),
        "targetPlacements": target_placements,
        "readyForOneReplica": provider_count >= int(manifest["shardCount"])
        and compatible_stages == int(manifest["shardCount"]),
        "readyForTargetReplication": provider_count >= target_placements
        and compatible_stages == int(manifest["shardCount"]),
        "additionalProvidersForOneReplica": max(
            0, int(manifest["shardCount"]) - provider_count
        ),
        "additionalProvidersForTargetReplication": max(
            0, target_placements - provider_count
        ),
    }
