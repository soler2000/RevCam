import pytest

from rev_cam.wifi import NMCLIBackend, WiFiError, WiFiStatus


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


def test_nmcli_start_hotspot_opens_network_without_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = NMCLIBackend(interface="wlan0")
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(list(args))
        if args[:3] == ["nmcli", "connection", "show"]:
            raise WiFiError("Unknown connection")
        return ""

    monkeypatch.setattr(backend, "_run", fake_run)
    monkeypatch.setattr(
        backend,
        "get_status",
        lambda: WiFiStatus(
            connected=False,
            ssid="RevCam",
            signal=None,
            ip_address=None,
            mode="access-point",
            hotspot_active=True,
            profile="RevCam Hotspot",
        ),
    )

    status = backend.start_hotspot("RevCam", None)

    assert status.hotspot_active is True
    assert [
        "nmcli",
        "connection",
        "modify",
        "RevCam Hotspot",
        "802-11-wireless-security.key-mgmt",
        "none",
    ] in commands
    assert [
        "nmcli",
        "connection",
        "modify",
        "RevCam Hotspot",
        "-802-11-wireless-security.psk",
    ] in commands
    assert "password" not in {item for command in commands for item in command}
    assert ["nmcli", "connection", "up", "RevCam Hotspot"] in commands


def test_nmcli_start_hotspot_with_password_uses_nmcli_hotspot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = NMCLIBackend(interface="wlan0")
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(list(args))
        return ""

    monkeypatch.setattr(backend, "_run", fake_run)
    monkeypatch.setattr(
        backend,
        "get_status",
        lambda: WiFiStatus(
            connected=False,
            ssid="RevCam",
            signal=None,
            ip_address=None,
            mode="access-point",
            hotspot_active=True,
            profile="RevCam Hotspot",
        ),
    )

    status = backend.start_hotspot("RevCam", "supersecret")

    assert status.hotspot_active is True
    assert commands[0] == [
        "nmcli",
        "device",
        "wifi",
        "hotspot",
        "ifname",
        "wlan0",
        "con-name",
        "RevCam Hotspot",
        "ssid",
        "RevCam",
        "password",
        "supersecret",
    ]
