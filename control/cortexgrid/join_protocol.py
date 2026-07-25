from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .util import canonical_json
from .wallet import Wallet, verify_signature


JOIN_PROTOCOL = "cortex-join-v1"
REGISTRATION_DOMAIN = "cortex-registration-v1"
ASSIGNMENT_DOMAIN = "cortex-assignment-v1"
STATUS_DOMAIN = "cortex-node-status-v1"
READY_DOMAIN = "cortex-node-ready-v1"
HEARTBEAT_DOMAIN = "cortex-node-heartbeat-v1"
LEAVE_DOMAIN = "cortex-node-leave-v1"
WORK_TICKET_DOMAIN = "cortex-work-ticket-v1"
INFERENCE_ACCESS_DOMAIN = "cortex-inference-access-v1"
MAX_CLOCK_SKEW_MS = 5 * 60 * 1_000


def now_ms() -> int:
    return int(time.time() * 1_000)


def signed_message(domain: str, payload: dict) -> str:
    return f"{domain}|{canonical_json(payload)}"


def sign_payload(wallet: Wallet, domain: str, payload: dict) -> str:
    return wallet.sign(signed_message(domain, payload))


def verify_payload(
    public_key: str, domain: str, payload: dict, signature: str
) -> bool:
    return verify_signature(public_key, signed_message(domain, payload), signature)


def require_fresh(timestamp_ms: object) -> int:
    try:
        timestamp = int(timestamp_ms)
    except (TypeError, ValueError) as error:
        raise ValueError("timestampMs must be an integer") from error
    if abs(now_ms() - timestamp) > MAX_CLOCK_SKEW_MS:
        raise ValueError("signed node message is outside the allowed clock window")
    return timestamp


def request_json(
    url: str,
    *,
    payload: dict | None = None,
    timeout: float = 10,
) -> dict:
    data = None if payload is None else canonical_json(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(detail).get("message", detail)
        except json.JSONDecodeError:
            message = detail
        raise RuntimeError(f"remote endpoint rejected the request: {message}") from error
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"remote endpoint is unavailable: {error}") from error
    if not isinstance(body, dict):
        raise RuntimeError("remote endpoint returned a non-object JSON response")
    return body
