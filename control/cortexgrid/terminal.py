from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field


DEFAULT_SERVER = "http://127.0.0.1:4173"
MAX_CONTEXT_CHARS = 7_000


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

    def infer(self, prompt: str, model_id: str | None = None) -> dict:
        payload = {"prompt": prompt}
        if model_id:
            payload["modelId"] = model_id
        return self.request("/api/inference", payload)


@dataclass
class CortexSession:
    api: CortexApi
    model_id: str | None = None
    show_metadata: bool = True
    color: bool = True
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
        result = self.api.infer(
            self.conversation_prompt(user_prompt), self.model_id
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
                    f"gateway paid {accounting.get('sponsorFee', accounting.get('fee', 0))} "
                    "CTX · end user paid 0 CTX",
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
/exit      leave CortexTerm
"""


def interactive(session: CortexSession) -> int:
    session.print_status()
    print("\nCortexTerm is wallet-free. Type a message or /help.\n")
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
    args = parser.parse_args()

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
    )
    if args.prompt is None:
        raise SystemExit(interactive(session))
    try:
        result = session.ask(args.prompt)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        session.print_result(result)


if __name__ == "__main__":
    main()
