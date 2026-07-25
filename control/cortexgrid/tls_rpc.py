from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import ipaddress
import os
import select
import socket
import ssl
import threading
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


RPC_TLS_PROTOCOL = "cortex-rpc-mtls-v1"


def certificate_fingerprint(pem: str | bytes) -> str:
    certificate = x509.load_pem_x509_certificate(
        pem.encode("utf-8") if isinstance(pem, str) else pem
    )
    return certificate.fingerprint(hashes.SHA256()).hex()


def _write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    path.chmod(0o600)


def _write_public(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o644)


def _private_pem(key: ed25519.Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _certificate_pem(certificate: x509.Certificate) -> bytes:
    return certificate.public_bytes(serialization.Encoding.PEM)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _san_value(host: str) -> x509.GeneralName:
    try:
        return x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        return x509.DNSName(host)


class ControllerTlsAuthority:
    """Controller-owned CA and the sole client identity accepted by providers."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.ca_key_path = data_dir / "ca-key.pem"
        self.ca_certificate_path = data_dir / "ca-certificate.pem"
        self.client_key_path = data_dir / "controller-client-key.pem"
        self.client_certificate_path = data_dir / "controller-client-certificate.pem"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not all(
            path.is_file()
            for path in (
                self.ca_key_path,
                self.ca_certificate_path,
                self.client_key_path,
                self.client_certificate_path,
            )
        ):
            self._create()
        self.ca_key_path.chmod(0o600)
        self.client_key_path.chmod(0o600)
        self.ca_key = serialization.load_pem_private_key(
            self.ca_key_path.read_bytes(), password=None
        )
        self.ca_certificate = x509.load_pem_x509_certificate(
            self.ca_certificate_path.read_bytes()
        )
        self.client_certificate = x509.load_pem_x509_certificate(
            self.client_certificate_path.read_bytes()
        )

    def _create(self) -> None:
        now = _utc_now()
        ca_key = ed25519.Ed25519PrivateKey.generate()
        ca_certificate = (
            x509.CertificateBuilder()
            .subject_name(_name("CortexGrid RPC Authority"))
            .issuer_name(_name("CortexGrid RPC Authority"))
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, algorithm=None)
        )
        client_key = ed25519.Ed25519PrivateKey.generate()
        client_certificate = (
            x509.CertificateBuilder()
            .subject_name(_name("CortexGrid Controller"))
            .issuer_name(ca_certificate.subject)
            .public_key(client_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=True,
            )
            .sign(ca_key, algorithm=None)
        )
        _write_private(self.ca_key_path, _private_pem(ca_key))
        _write_public(self.ca_certificate_path, _certificate_pem(ca_certificate))
        _write_private(self.client_key_path, _private_pem(client_key))
        _write_public(
            self.client_certificate_path, _certificate_pem(client_certificate)
        )

    @property
    def ca_pem(self) -> str:
        return self.ca_certificate_path.read_text(encoding="utf-8")

    @property
    def client_fingerprint(self) -> str:
        return self.client_certificate.fingerprint(hashes.SHA256()).hex()

    def issue_server_certificate(
        self, *, csr_pem: str, node_id: str, advertise_host: str
    ) -> str:
        csr = x509.load_pem_x509_csr(csr_pem.encode("utf-8"))
        if not csr.is_signature_valid:
            raise ValueError("RPC certificate request signature is invalid")
        now = _utc_now()
        certificate = (
            x509.CertificateBuilder()
            .subject_name(_name(f"CortexGrid Provider {node_id}"))
            .issuer_name(self.ca_certificate.subject)
            .public_key(csr.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=90))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName(
                    [_san_value(advertise_host), x509.DNSName(node_id)]
                ),
                critical=False,
            )
            .sign(self.ca_key, algorithm=None)
        )
        return _certificate_pem(certificate).decode("utf-8")


def create_provider_csr(
    *, key_path: Path, node_id: str, advertise_host: str
) -> str:
    if key_path.exists():
        key_path.chmod(0o600)
        key = serialization.load_pem_private_key(
            key_path.read_bytes(), password=None
        )
    else:
        key = ed25519.Ed25519PrivateKey.generate()
        _write_private(key_path, _private_pem(key))
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(_name(f"CortexGrid Provider {node_id}"))
        .add_extension(
            x509.SubjectAlternativeName(
                [_san_value(advertise_host), x509.DNSName(node_id)]
            ),
            critical=False,
        )
        .sign(key, algorithm=None)
    )
    return csr.public_bytes(serialization.Encoding.PEM).decode("utf-8")


def _bridge(left: socket.socket, right: socket.socket) -> None:
    peers = {left: right, right: left}
    outgoing = {left: bytearray(), right: bytearray()}
    readable = {left: True, right: True}
    write_shutdown = {left: False, right: False}
    read_needs_write: set[socket.socket] = set()
    send_needs_read: set[socket.socket] = set()
    sockets = (left, right)
    for connection in sockets:
        connection.setblocking(False)

    def receive(source: socket.socket) -> bool:
        try:
            payload = source.recv(1024 * 1024)
        except ssl.SSLWantReadError:
            return True
        except ssl.SSLWantWriteError:
            read_needs_write.add(source)
            return True
        except (OSError, ssl.SSLError):
            payload = b""
        if payload:
            outgoing[peers[source]].extend(payload)
        else:
            readable[source] = False
        return True

    def transmit(target: socket.socket) -> bool:
        if not outgoing[target]:
            return True
        try:
            sent = target.send(outgoing[target])
        except ssl.SSLWantWriteError:
            return True
        except ssl.SSLWantReadError:
            send_needs_read.add(target)
            return True
        except (OSError, ssl.SSLError):
            return False
        del outgoing[target][:sent]
        return True

    try:
        while any(readable.values()) or any(outgoing.values()):
            read_wait = [
                connection
                for connection in sockets
                if (
                    readable[connection]
                    and connection not in read_needs_write
                    and len(outgoing[peers[connection]]) < 16 * 1024 * 1024
                )
                or connection in send_needs_read
            ]
            write_wait = [
                connection
                for connection in sockets
                if (
                    outgoing[connection]
                    and connection not in send_needs_read
                )
                or connection in read_needs_write
            ]
            if not read_wait and not write_wait:
                break
            ready_read, ready_write, exceptional = select.select(
                read_wait, write_wait, list(sockets), 1
            )
            if exceptional:
                break
            for connection in ready_read:
                if connection in send_needs_read:
                    send_needs_read.discard(connection)
                    if not transmit(connection):
                        raise OSError("RPC bridge send failed")
                if readable[connection] and connection not in read_needs_write:
                    receive(connection)
            for connection in ready_write:
                if connection in read_needs_write:
                    read_needs_write.discard(connection)
                    receive(connection)
                if outgoing[connection] and connection not in send_needs_read:
                    if not transmit(connection):
                        raise OSError("RPC bridge send failed")
            for source in sockets:
                target = peers[source]
                if (
                    not readable[source]
                    and not outgoing[target]
                    and not write_shutdown[target]
                ):
                    write_shutdown[target] = True
                    try:
                        target.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
    finally:
        for connection in sockets:
            try:
                connection.close()
            except OSError:
                pass


class _Forwarder:
    def __init__(self) -> None:
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def port(self) -> int:
        if self._listener is None:
            raise RuntimeError("RPC forwarder is not running")
        return int(self._listener.getsockname()[1])

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._listener.close()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class TlsRpcServer(_Forwarder):
    """Provider-side mTLS listener forwarding to loopback-only llama.cpp RPC."""

    def __init__(
        self,
        *,
        listen_host: str,
        listen_port: int,
        upstream_host: str,
        upstream_port: int,
        certificate_path: Path,
        key_path: Path,
        ca_certificate_path: Path,
        allowed_client_fingerprint: str,
    ):
        super().__init__()
        self.upstream = (upstream_host, upstream_port)
        self.allowed_client_fingerprint = allowed_client_fingerprint
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_cert_chain(certificate_path, key_path)
        context.load_verify_locations(cafile=str(ca_certificate_path))
        self.context = context
        listener = socket.socket(socket.AF_INET6 if ":" in listen_host else socket.AF_INET)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listen_host, listen_port))
        listener.listen(32)
        listener.settimeout(0.5)
        self._listener = listener
        self._thread = threading.Thread(
            target=self._serve, name="cortex-rpc-mtls-server", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except (OSError, TimeoutError):
                continue
            threading.Thread(
                target=self._handle, args=(connection,), daemon=True
            ).start()

    def _handle(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(5)
            secure = self.context.wrap_socket(connection, server_side=True)
            peer = secure.getpeercert(binary_form=True)
            if (
                peer is None
                or not hmac.compare_digest(
                    hashlib.sha256(peer).hexdigest(),
                    self.allowed_client_fingerprint,
                )
            ):
                secure.close()
                return
            upstream = socket.create_connection(self.upstream, timeout=10)
            _bridge(secure, upstream)
        except (OSError, ssl.SSLError):
            connection.close()


class TlsRpcClient(_Forwarder):
    """Controller-side loopback proxy presenting the sole authorized client cert."""

    def __init__(
        self,
        *,
        target_host: str,
        target_port: int,
        expected_server_fingerprint: str,
        certificate_path: Path,
        key_path: Path,
        ca_certificate_path: Path,
    ):
        super().__init__()
        self.target = (target_host, target_port)
        self.expected_server_fingerprint = expected_server_fingerprint
        context = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH, cafile=str(ca_certificate_path)
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(certificate_path, key_path)
        self.context = context
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(32)
        listener.settimeout(0.5)
        self._listener = listener
        self._thread = threading.Thread(
            target=self._serve, name="cortex-rpc-mtls-client", daemon=True
        )
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return f"127.0.0.1:{self.port}"

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except (OSError, TimeoutError):
                continue
            threading.Thread(
                target=self._handle, args=(connection,), daemon=True
            ).start()

    def _handle(self, connection: socket.socket) -> None:
        try:
            raw = socket.create_connection(self.target, timeout=10)
            secure = self.context.wrap_socket(
                raw, server_hostname=self.target[0]
            )
            peer = secure.getpeercert(binary_form=True)
            if (
                peer is None
                or not hmac.compare_digest(
                    hashlib.sha256(peer).hexdigest(),
                    self.expected_server_fingerprint,
                )
            ):
                secure.close()
                connection.close()
                return
            _bridge(connection, secure)
        except (OSError, ssl.SSLError):
            connection.close()

    def probe(self, timeout: float = 3) -> None:
        raw = socket.create_connection(self.target, timeout=timeout)
        try:
            secure = self.context.wrap_socket(
                raw, server_hostname=self.target[0]
            )
            peer = secure.getpeercert(binary_form=True)
            if (
                peer is None
                or not hmac.compare_digest(
                    hashlib.sha256(peer).hexdigest(),
                    self.expected_server_fingerprint,
                )
            ):
                raise ssl.SSLError("provider RPC certificate fingerprint mismatch")
            secure.close()
        finally:
            raw.close()
