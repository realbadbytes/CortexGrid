from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import secrets
import socket
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .economics import DEFAULT_PRICE_MICRO_PER_WORK_UNIT
from .node_daemon import ProviderNode


DEFAULT_SERVER = "http://127.0.0.1:4173"
MAX_CONTEXT_CHARS = 7_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CortexApi:
    def __init__(self, server: str, timeout: float = 180):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
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
            raise RuntimeError(f"CortexGrid rejected the request: {message}") from error
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(
                f"Cannot reach CortexGrid at {self.server}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise RuntimeError("CortexGrid returned invalid JSON") from error
        if not isinstance(result, dict):
            raise RuntimeError("CortexGrid returned an invalid response")
        return result

    def network(self) -> dict:
        return self.request("/api/network")

    def infer(
        self,
        prompt: str,
        model_id: str | None,
        participation: dict,
    ) -> dict:
        payload = {"prompt": prompt, "participation": participation}
        if model_id:
            payload["modelId"] = model_id
        return self.request("/api/inference", payload)


@dataclass
class CortexSession:
    api: CortexApi
    model_id: str | None = None
    show_metadata: bool = True
    color: bool = True
    participation_signer: (
        Callable[[str, str | None], dict] | None
    ) = None
    participant_status: Callable[[], dict] | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    last_result: dict | None = None

    def _paint(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def conversation_prompt(self, user_prompt: str) -> str:
        if not self.history:
            return user_prompt
        sections = [
            "Continue the following conversation. Answer the final user message.",
        ]
        for user, assistant in self.history:
            sections.extend((f"User: {user}", f"Assistant: {assistant}"))
        sections.append(f"User: {user_prompt}")
        sections.append("Assistant:")
        prompt = "\n".join(sections)
        if len(prompt) <= MAX_CONTEXT_CHARS:
            return prompt

        recent: list[tuple[str, str]] = []
        size = len(user_prompt) + 100
        for exchange in reversed(self.history):
            exchange_size = len(exchange[0]) + len(exchange[1]) + 20
            if size + exchange_size > MAX_CONTEXT_CHARS:
                break
            recent.append(exchange)
            size += exchange_size
        recent.reverse()
        sections = [
            "Continue the following conversation. Answer the final user message.",
        ]
        for user, assistant in recent:
            sections.extend((f"User: {user}", f"Assistant: {assistant}"))
        sections.extend((f"User: {user_prompt}", "Assistant:"))
        return "\n".join(sections)

    @staticmethod
    def route_summary(result: dict) -> str:
        route = result.get("route", [])
        return " → ".join(
            f"{stage.get('nodeName', stage['nodeId'])}"
            f"[{stage.get('stage', '?')}]"
            for stage in route
        )

    def ask(self, user_prompt: str) -> dict:
        if self.participation_signer is None:
            raise RuntimeError(
                "CortexTerm must join as a compute participant before inference"
            )
        prompt = self.conversation_prompt(user_prompt)
        result = self.api.infer(
            prompt,
            self.model_id,
            self.participation_signer(prompt, self.model_id),
        )
        answer = result["output"]["text"]
        self.history.append((user_prompt, answer))
        self.last_result = result
        return result

    def print_result(self, result: dict) -> None:
        print()
        print(self._paint("1;36", "Cortex"))
        print(textwrap.fill(result["output"]["text"], width=88))
        if self.show_metadata:
            block = result.get("block", {})
            accounting = result.get("accounting", {})
            proof = result.get("proof", {})
            print()
            print(
                self._paint("2", f"route  {self.route_summary(result)}")
            )
            print(
                self._paint(
                    "2",
                    f"block  #{block.get('height', '?')} · "
                    f"{str(block.get('hash', ''))[:16]}… · "
                    f"proof {str(proof.get('root', ''))[:16]}…",
                )
            )
            print(
                self._paint(
                    "2",
                    f"reward {accounting.get('providerRewards', 0)} "
                    f"{accounting.get('symbol', 'CTX')} to compute providers · "
                    "access earned by online compute participation · "
                    "requester paid 0 CTX",
                )
            )
        print()

    def print_status(self) -> None:
        network = self.api.network()
        metrics = network["metrics"]
        model = network["model"]
        print(
            f"{network['name']} {network['environment']} · {network['status']}\n"
            f"Model: {model['name']} ({model['precision']})\n"
            f"Providers: {metrics['nodesOnline']}/{metrics['nodesTotal']} online · "
            f"Stages: {model['shardCount']} · Chain: #{metrics['chainHeight']}"
        )
        if self.participant_status is not None:
            participant = self.participant_status()
            assignment = participant.get("assignment") or {}
            shard = assignment.get("shard") or {}
            stage = shard.get("index", "pending")
            print(
                f"You: {participant['nodeId']} · {participant['role']} · "
                f"stage {stage} · {participant['balance']} CTX earned"
            )

    def print_last_proof(self) -> None:
        if self.last_result is None:
            print("No inference has been run in this session.")
            return
        result = self.last_result
        print(
            json.dumps(
                {
                    "inferenceId": result["id"],
                    "route": result["route"],
                    "proofRoot": result["proof"]["root"],
                    "block": result["block"],
                    "accounting": result["accounting"],
                },
                indent=2,
            )
        )


HELP = """\
/new       clear conversation context
/status    show network and model status
/route     show the last distributed route
/proof     show the last proof, block, and provider rewards
/metadata  toggle route/block metadata
/help      show commands
/exit      stop serving, leave the network, and exit
"""


def interactive(session: CortexSession) -> int:
    session.print_status()
    print(
        "\nCortexTerm is online as a compute participant. "
        "Type a message or /help.\n"
    )
    while True:
        try:
            user_prompt = input(session._paint("1;32", "you > ")).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nUse /exit to quit.")
            continue
        if not user_prompt:
            continue
        if user_prompt == "/exit":
            return 0
        if user_prompt == "/help":
            print(HELP)
            continue
        if user_prompt == "/new":
            session.history.clear()
            print("Conversation cleared.")
            continue
        if user_prompt == "/status":
            try:
                session.print_status()
            except RuntimeError as error:
                print(error, file=sys.stderr)
            continue
        if user_prompt == "/route":
            if session.last_result is None:
                print("No route yet.")
            else:
                print(session.route_summary(session.last_result))
            continue
        if user_prompt == "/proof":
            session.print_last_proof()
            continue
        if user_prompt == "/metadata":
            session.show_metadata = not session.show_metadata
            print(f"Metadata {'on' if session.show_metadata else 'off'}.")
            continue
        if user_prompt.startswith("/"):
            print("Unknown command. Use /help.")
            continue
        try:
            print(session._paint("2", "running distributed inference…"))
            session.print_result(session.ask(user_prompt))
        except RuntimeError as error:
            print(error, file=sys.stderr)


def detect_available_memory_gib() -> float:
    """Advertise most, but not all, currently available system memory."""
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return round(max(0.5, pages * page_size / (1024**3) * 0.8), 2)
    except (AttributeError, OSError, TypeError, ValueError):
        return 8.0


def load_or_create_node_id(data_dir: Path) -> str:
    path = data_dir / "node_id"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    hostname = re.sub(
        r"[^a-z0-9]+", "-", socket.gethostname().lower()
    ).strip("-")
    hostname = hostname[:28] or "host"
    node_id = f"term-{hostname}-{secrets.token_hex(3)}"
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(node_id + "\n", encoding="utf-8")
    path.chmod(0o600)
    return node_id


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="CortexTerm",
        description="Chat directly with the CortexGrid distributed inference network",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("CORTEX_URL", DEFAULT_SERVER),
        help="CortexGrid coordinator URL",
    )
    parser.add_argument("--model", help="model id; defaults to the network model")
    parser.add_argument("-p", "--prompt", help="run one prompt and exit")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="hide route, proof, block, and reward summary",
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--node-id")
    parser.add_argument("--name", help="compute participant display name")
    parser.add_argument("--operator", default=getpass.getuser())
    parser.add_argument("--region", default="unknown")
    parser.add_argument(
        "--memory-gib",
        type=float,
        help="RAM/VRAM to offer; defaults to 80%% of currently available RAM",
    )
    parser.add_argument("--latency-ms", type=int, default=50)
    parser.add_argument(
        "--price-micro-per-work-unit",
        type=int,
        default=DEFAULT_PRICE_MICRO_PER_WORK_UNIT,
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument(
        "--advertise-host",
        default="127.0.0.1",
        help="address the coordinator can use to reach this machine",
    )
    parser.add_argument("--provider-api-port", type=int, default=0)
    parser.add_argument("--provider-rpc-port", type=int, default=0)
    parser.add_argument(
        "--executor",
        type=Path,
        default=PROJECT_ROOT / "build" / "cortex_worker",
    )
    parser.add_argument(
        "--rpc-binary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "build"
            / "llama"
            / "bin"
            / "ggml-rpc-server"
        ),
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--coordinator-public-key")
    args = parser.parse_args()

    data_dir = args.data_dir or PROJECT_ROOT / ".provider" / "cortexterm"
    node_id = args.node_id or load_or_create_node_id(data_dir)
    if args.node_id and args.data_dir is None:
        data_dir = PROJECT_ROOT / ".provider" / args.node_id
    provider = ProviderNode(
        coordinator_url=args.server,
        node_id=node_id,
        name=args.name or f"CortexTerm on {socket.gethostname()}",
        operator=args.operator,
        region=args.region,
        memory_gib=(
            args.memory_gib
            if args.memory_gib is not None
            else detect_available_memory_gib()
        ),
        network_latency_ms=args.latency_ms,
        executor_path=args.executor,
        rpc_binary=args.rpc_binary,
        data_dir=data_dir,
        listen_host=args.listen_host,
        advertise_host=args.advertise_host,
        api_port=args.provider_api_port,
        rpc_port=args.provider_rpc_port,
        coordinator_public_key=args.coordinator_public_key,
        price_micro_per_work_unit=args.price_micro_per_work_unit,
        role="interactive-user",
    )
    try:
        joined = provider.start()
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        provider.close()
        print(
            f"CortexTerm could not join as a compute participant: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    assignment = joined["assignment"]["shard"]
    print(
        f"Serving stage {assignment['index']} as {node_id}; "
        f"rewards go to {provider.reward_address}.",
        file=sys.stderr,
    )

    color = (
        sys.stdout.isatty()
        and not args.no_color
        and "NO_COLOR" not in os.environ
    )
    session = CortexSession(
        CortexApi(args.server),
        model_id=args.model,
        show_metadata=not args.no_metadata,
        color=color,
        participation_signer=provider.authorize_inference,
        participant_status=provider.snapshot,
    )
    try:
        if args.prompt is None:
            code = interactive(session)
        else:
            try:
                result = session.ask(args.prompt)
            except RuntimeError as error:
                print(error, file=sys.stderr)
                raise SystemExit(1) from error
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                session.print_result(result)
            code = 0
    finally:
        if provider.joined:
            try:
                provider.leave_network()
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
                print(f"CortexTerm leave failed: {error}", file=sys.stderr)
        provider.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
