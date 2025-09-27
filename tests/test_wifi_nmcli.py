import pytest

from rev_cam.wifi import NMCLIBackend, WiFiStatus


def _status(ssid: str) -> WiFiStatus:
    return WiFiStatus(
        connected=True,
        ssid=ssid,
        signal=70,
        ip_address="192.168.1.2",
        mode="station",
        hotspot_active=False,
        profile=ssid,
        detail=f"Connected to {ssid}",
    )


def test_nmcli_connect_uses_saved_profile_when_no_password(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NMCLIBackend(interface="wlan0")
    monkeypatch.setattr(backend, "_parse_saved_profiles", lambda: {"Home"})
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(list(args))
        return ""

    monkeypatch.setattr(backend, "_run", fake_run)
    monkeypatch.setattr(backend, "get_status", lambda: _status("Home"))

    status = backend.connect("Home", None)

    assert status.ssid == "Home"
    assert commands == [["nmcli", "connection", "up", "Home", "ifname", "wlan0"]]


def test_nmcli_connect_uses_password_for_new_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NMCLIBackend(interface="wlan0")
    monkeypatch.setattr(backend, "_parse_saved_profiles", lambda: {"Home"})
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(list(args))
        return ""

    monkeypatch.setattr(backend, "_run", fake_run)
    monkeypatch.setattr(backend, "get_status", lambda: _status("Home"))

    status = backend.connect("Home", "supersecret")

    assert status.ssid == "Home"
    assert commands == [
        [
            "nmcli",
            "device",
            "wifi",
            "connect",
            "Home",
            "password",
            "supersecret",
            "ifname",
            "wlan0",
        ]
    ]
