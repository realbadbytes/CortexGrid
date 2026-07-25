from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .util import sha256


def address_from_public_key(public_key_hex: str) -> str:
    return f"ctx_{sha256(bytes.fromhex(public_key_hex))[:40]}"


def transfer_message(
    from_address: str, to_address: str, amount_micro: int, nonce: int
) -> str:
    return (
        f"ctx-transfer-v1|{from_address}|{to_address}|{amount_micro}|{nonce}"
    )


def verify_signature(public_key_hex: str, message: str, signature_hex: str) -> bool:
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        public_key.verify(bytes.fromhex(signature_hex), message.encode("utf-8"))
        return True
    except (ValueError, TypeError):
        return False
    except Exception:
        return False


@dataclass(frozen=True)
class Wallet:
    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls) -> "Wallet":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: Path) -> "Wallet":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            private_key = Ed25519PrivateKey.from_private_bytes(
                bytes.fromhex(data["privateKey"])
            )
            return cls(private_key)

        wallet = cls.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"privateKey": wallet.private_key_hex}, indent=2) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o600)
        return wallet

    @property
    def private_key_hex(self) -> str:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        ).hex()

    @property
    def public_key_hex(self) -> str:
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

    @property
    def address(self) -> str:
        return address_from_public_key(self.public_key_hex)

    def sign(self, message: str) -> str:
        return self.private_key.sign(message.encode("utf-8")).hex()

    def sign_transfer(self, to_address: str, amount_micro: int, nonce: int) -> dict:
        message = transfer_message(self.address, to_address, amount_micro, nonce)
        return {
            "from": self.address,
            "to": to_address,
            "amountMicro": amount_micro,
            "nonce": nonce,
            "publicKey": self.public_key_hex,
            "signature": self.sign(message),
        }
