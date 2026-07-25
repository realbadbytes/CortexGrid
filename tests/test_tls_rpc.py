from __future__ import annotations

import socket
import ssl
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from cortexgrid.tls_rpc import (
    ControllerTlsAuthority,
    TlsRpcClient,
    TlsRpcServer,
    certificate_fingerprint,
    create_provider_csr,
)


class _EchoServer:
    def __init__(self) -> None:
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.listener.settimeout(0.2)
        self.port = int(self.listener.getsockname()[1])
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while not self.stop.is_set():
            try:
                connection, _ = self.listener.accept()
            except (OSError, TimeoutError):
                continue
            threading.Thread(
                target=self._echo, args=(connection,), daemon=True
            ).start()

    @staticmethod
    def _echo(connection: socket.socket) -> None:
        with connection:
            try:
                payload = connection.recv(4096)
                if payload:
                    connection.sendall(payload)
            except OSError:
                pass

    def close(self) -> None:
        self.stop.set()
        self.listener.close()
        self.thread.join(timeout=2)


class ControllerOnlyTlsRpcTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.authority = ControllerTlsAuthority(root / "authority")
        provider_key = root / "provider-key.pem"
        certificate = self.authority.issue_server_certificate(
            csr_pem=create_provider_csr(
                key_path=provider_key,
                node_id="provider-test",
                advertise_host="127.0.0.1",
            ),
            node_id="provider-test",
            advertise_host="127.0.0.1",
        )
        provider_certificate = root / "provider-certificate.pem"
        provider_certificate.write_text(certificate, encoding="utf-8")
        self.echo = _EchoServer()
        self.server = TlsRpcServer(
            listen_host="127.0.0.1",
            listen_port=0,
            upstream_host="127.0.0.1",
            upstream_port=self.echo.port,
            certificate_path=provider_certificate,
            key_path=provider_key,
            ca_certificate_path=self.authority.ca_certificate_path,
            allowed_client_fingerprint=self.authority.client_fingerprint,
        )
        self.client = TlsRpcClient(
            target_host="127.0.0.1",
            target_port=self.server.port,
            expected_server_fingerprint=certificate_fingerprint(certificate),
            certificate_path=self.authority.client_certificate_path,
            key_path=self.authority.client_key_path,
            ca_certificate_path=self.authority.ca_certificate_path,
        )

    def tearDown(self) -> None:
        self.client.close()
        self.server.close()
        self.echo.close()
        self.temporary_directory.cleanup()

    def test_controller_certificate_can_reach_private_rpc(self) -> None:
        with socket.create_connection(("127.0.0.1", self.client.port)) as connection:
            connection.sendall(b"controller-only")
            self.assertEqual(connection.recv(4096), b"controller-only")

    def test_controller_private_keys_are_owner_only(self) -> None:
        self.assertEqual(
            stat.S_IMODE(self.authority.ca_key_path.stat().st_mode), 0o600
        )
        self.assertEqual(
            stat.S_IMODE(self.authority.client_key_path.stat().st_mode), 0o600
        )

    def test_plaintext_client_is_rejected_before_upstream(self) -> None:
        with socket.create_connection(("127.0.0.1", self.server.port)) as connection:
            connection.settimeout(2)
            connection.sendall(b"not tls")
            try:
                response = connection.recv(4096)
            except (ConnectionResetError, TimeoutError):
                response = b""
            self.assertEqual(response, b"")

    def test_client_without_controller_certificate_is_rejected(self) -> None:
        context = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH,
            cafile=str(self.authority.ca_certificate_path),
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        raw = socket.create_connection(("127.0.0.1", self.server.port))
        secure: ssl.SSLSocket | None = None
        try:
            with self.assertRaises((ssl.SSLError, ConnectionError, OSError)):
                secure = context.wrap_socket(
                    raw, server_hostname="127.0.0.1"
                )
                secure.sendall(b"unauthorized")
                if not secure.recv(1):
                    raise ConnectionError("provider rejected unauthenticated RPC")
        finally:
            if secure is not None:
                secure.close()
            else:
                raw.close()


if __name__ == "__main__":
    unittest.main()
