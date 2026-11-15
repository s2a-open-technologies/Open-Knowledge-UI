import socket

import pytest

from app.core import netguard


def _resolves_to(monkeypatch, ip):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda host, port: [(socket.AF_INET, 0, 0, "", (ip, 0))]
    )


def test_link_local_ip_and_resolved_hostname_are_blocked(monkeypatch):
    with pytest.raises(netguard.BlockedTargetError, match="Link-Local"):
        netguard.reject_link_local("http://169.254.169.254/latest")
    _resolves_to(monkeypatch, "169.254.169.254")
    with pytest.raises(netguard.BlockedTargetError):
        netguard.reject_link_local("https://metadata.intern/x")
    with pytest.raises(ValueError):  # Hostnamen ohne URL (SQL-Host)
        netguard.reject_host_link_local("metadata.intern")


def test_private_and_unresolvable_targets_stay_allowed(monkeypatch):
    _resolves_to(monkeypatch, "192.168.1.10")
    netguard.reject_link_local("http://vllm.lan:8000/v1")

    def boom(host, port):
        raise socket.gaierror("kein DNS")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    netguard.reject_link_local("http://gibt-es-nicht.example")


def test_the_callers_error_type_is_raised_and_a_missing_host_is_refused():
    class MyError(Exception):
        pass

    with pytest.raises(MyError, match="Link-Local"):
        netguard.reject_link_local("http://169.254.1.1", MyError)
    with pytest.raises(MyError, match="Ungültige"):
        netguard.reject_link_local("kein-host", MyError)


@pytest.mark.parametrize(
    "address",
    [
        "::ffff:169.254.169.254",  # IPv4-gemappt
        "64:ff9b::a9fe:a9fe",  # NAT64
        "fd00:ec2::254",  # AWS (IPv6)
        "100.100.100.200",  # Alibaba Cloud
        "fe80::1%eth0",
    ],
)
def test_metadata_addresses_are_blocked_in_every_form(monkeypatch, address):
    _resolves_to(monkeypatch, address)
    with pytest.raises(netguard.BlockedTargetError):
        netguard.reject_link_local("http://irgendwas.intern/")
