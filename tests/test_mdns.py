import asyncio
import logging
import threading
import types

import pytest

from rev_cam import mdns


class _FakeServiceInfo:
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple stub
        self.args = args
        self.kwargs = kwargs


class _FailingZeroconf:
    instances: list["_FailingZeroconf"] = []

    def __init__(self, interfaces=None) -> None:  # noqa: D401 - simple stub
        self.interfaces = interfaces
        self.closed = False
        self.register_calls = 0
        _FailingZeroconf.instances.append(self)

    def register_service(self, info, allow_name_change=False) -> None:  # noqa: D401
        del info, allow_name_change
        self.register_calls += 1
        raise PermissionError("Can't open /dev/mem: Permission denied")

    def unregister_service(self, info) -> None:  # noqa: D401 - simple stub
        del info

    def close(self) -> None:  # noqa: D401 - simple stub
        self.closed = True


def test_mdns_permission_error_disables_advertising(monkeypatch, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="rev_cam.mdns")

    _FailingZeroconf.instances.clear()

    monkeypatch.setattr(mdns, "ServiceInfo", _FakeServiceInfo)
    monkeypatch.setattr(mdns, "Zeroconf", _FailingZeroconf)
    monkeypatch.setattr(mdns, "InterfaceChoice", types.SimpleNamespace(All="all"))

    advertiser = mdns.MDNSAdvertiser(hostname="test.local")

    advertiser.advertise("192.168.0.2")

    assert "mDNS advertising disabled" in caplog.text
    assert "/dev/mem" in caplog.text or "sudo" in caplog.text

    backend = advertiser._backend  # noqa: SLF001 - validating internal guard
    assert getattr(backend, "_disabled_reason") is not None

    caplog.clear()
    advertiser.advertise("192.168.0.3")
    assert caplog.records == []

    assert len(_FailingZeroconf.instances) == 1
    assert _FailingZeroconf.instances[0].register_calls == 1


def test_mdns_zeroconf_constructed_off_event_loop(monkeypatch) -> None:
    created_threads: list[int] = []
    main_thread = threading.get_ident()

    class _ThreadRecordingZeroconf:
        def __init__(self, interfaces=None) -> None:  # noqa: D401 - simple stub
            created_threads.append(threading.get_ident())

        def register_service(self, info, allow_name_change=False) -> None:  # noqa: D401
            del info, allow_name_change

        def unregister_service(self, info) -> None:  # noqa: D401 - simple stub
            del info

        def close(self) -> None:  # noqa: D401 - simple stub
            return

    monkeypatch.setattr(mdns, "Zeroconf", _ThreadRecordingZeroconf)
    monkeypatch.setattr(mdns, "ServiceInfo", _FakeServiceInfo)
    monkeypatch.setattr(mdns, "InterfaceChoice", types.SimpleNamespace(All="all"))

    advertiser = mdns.MDNSAdvertiser(hostname="test.local")

    async def _invoke() -> None:
        advertiser.advertise("192.168.1.2")

    asyncio.run(_invoke())

    assert created_threads, "Zeroconf constructor was not called"
    # Ensure the Zeroconf instance was created on a different thread while the
    # event loop was running, preventing EventLoopBlocked errors.
    assert created_threads[0] != main_thread
