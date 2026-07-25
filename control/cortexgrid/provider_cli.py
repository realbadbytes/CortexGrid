from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProviderOperatorApi:
    def __init__(self, endpoint: str, token: str, timeout: float = 15):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, path: str, *, post: bool = False) -> dict:
        data = b"{}" if post else None
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                **({"Content-Type": "application/json"} if post else {}),
            },
            method="POST" if post else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(detail).get("message", detail)
            except json.JSONDecodeError:
                message = detail
            raise RuntimeError(f"provider rejected the request: {message}") from error
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(
                f"cannot reach provider daemon at {self.endpoint}: {error}"
            ) from error
        if not isinstance(result, dict):
            raise RuntimeError("provider daemon returned an invalid response")
        return result


def format_status(status: dict) -> str:
    assignment = status.get("assignment") or {}
    model = assignment.get("model", {})
    shard = assignment.get("shard", {})
    layers = shard.get("layerRange")
    layer_text = (
        f"{layers[0]}–{layers[1]}"
        if isinstance(layers, list) and len(layers) == 2
        else "n/a"
    )
    state = "joined" if status.get("joined") else "left"
    rpc_state = "online" if status.get("rpcOnline") else "offline"
    rpc_transport = status.get("rpcTransport", "unknown")
    return (
        f"{status['name']} ({status['nodeId']}) · {state}\n"
        f"Operator: {status['operator']} · Region: {status['region']}\n"
        f"RPC: {rpc_state} at {status['rpcEndpoint']} · {rpc_transport}\n"
        f"Model: {model.get('name', 'unassigned')} "
        f"{model.get('version', '')}\n"
        f"Stage: {shard.get('index', 'n/a')} · Layers: {layer_text} · "
        f"Weight: {str(shard.get('weightsDigest', ''))[:16]}…\n"
        f"Jobs: {len(status.get('activeJobs', []))} active · "
        f"{status.get('completedJobs', 0)} completed\n"
        f"Bid: {status.get('priceMicroPerWorkUnit', 'n/a')} micro-CTX/work unit\n"
        f"CTX: {status.get('balance', 0)} · "
        f"{status.get('rewardAddress', '')}\n"
        f"Chain: #{status.get('chainHeight', 0)} · "
        f"Heartbeat: {status.get('lastHeartbeatAt') or 'not yet'}"
        + (
            f"\nLast error: {status['lastError']}"
            if status.get("lastError")
            else ""
        )
    )


def format_jobs(jobs: dict) -> str:
    lines: list[str] = []
    active = jobs.get("active", [])
    recent = jobs.get("recent", [])
    lines.append(f"Active jobs ({len(active)})")
    if not active:
        lines.append("  none")
    for job in active:
        lines.append(
            f"  {job['requestId']} · stage {job.get('stage')} · "
            f"{job.get('status')} · since {job.get('startedAt')}"
        )
    lines.append(f"Recent jobs ({len(recent)})")
    if not recent:
        lines.append("  none")
    for job in recent[:20]:
        reward = (
            f" · {job.get('reward')} CTX"
            if "reward" in job
            else ""
        )
        block = (
            f" · block #{job.get('blockHeight')}"
            if "blockHeight" in job
            else ""
        )
        lines.append(
            f"  {job['requestId']} · {job.get('status')} · "
            f"stage {job.get('stage', '?')}{reward}{block}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="CortexNode",
        description="Operate a local CortexGrid compute provider",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=("status", "jobs", "balance", "assignment", "join", "leave", "watch"),
    )
    parser.add_argument("--node-id", default="provider-one")
    parser.add_argument("--api", default="http://127.0.0.1:51800")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--token")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()

    data_dir = args.data_dir or PROJECT_ROOT / ".provider" / args.node_id
    token_path = data_dir / "operator_token"
    token = args.token
    if token is None:
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            print(
                f"Cannot read {token_path}. Start the provider daemon first or use --token.",
                file=sys.stderr,
            )
            raise SystemExit(1) from error
    api = ProviderOperatorApi(args.api, token)

    def fetch() -> dict:
        if args.command == "jobs":
            return api.request("/v1/jobs")
        if args.command == "join":
            return api.request("/v1/operator/join", post=True)
        if args.command == "leave":
            return api.request("/v1/operator/leave", post=True)
        return api.request("/v1/provider")

    try:
        if args.command == "watch":
            while True:
                status = api.request("/v1/provider")
                print("\033[2J\033[H" + format_status(status), flush=True)
                time.sleep(max(0.25, args.interval))
        result = fetch()
    except KeyboardInterrupt:
        print()
        return
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    if args.json:
        print(json.dumps(result, indent=2))
    elif args.command == "jobs":
        print(format_jobs(result))
    elif args.command == "balance":
        print(f"{result['balance']} CTX\n{result['rewardAddress']}")
    elif args.command == "assignment":
        print(json.dumps(result.get("assignment"), indent=2))
    elif args.command in {"join", "leave"}:
        print(
            f"{result['status']} · {result.get('balance', 0)} CTX · "
            f"{result.get('rewardAddress', '')}"
        )
    else:
        print(format_status(result))


if __name__ == "__main__":
    main()
