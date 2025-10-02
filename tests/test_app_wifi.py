from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam.app import create_app
from rev_cam.wifi import (
    DEFAULT_HOTSPOT_PASSWORD,
    WiFiCredentialStore,
    WiFiError,
    WiFiManager,
    WiFiNetwork,
    WiFiStatus,
)


class FakeWiFiBackend:
    def __init__(self) -> None:
        self.connect_attempts: list[tuple[str, str | None]] = []
        self.status = WiFiStatus(
            connected=True,
            ssid="Home",
            signal=75,
            ip_address="192.168.1.10",
            mode="station",
            hotspot_active=False,
            profile="Home",
            detail="Connected",
        )
        self.networks = [
            WiFiNetwork(
                ssid="Home",
                signal=75,
                security="WPA2",
                frequency=2412.0,
                channel=1,
                known=True,
                active=True,
            ),
            WiFiNetwork(
                ssid="Guest",
                signal=55,
                security="WPA2",
                frequency=2417.0,
                channel=2,
                known=False,
                active=False,
            ),
        ]
        self.hotspot_error: str | None = None
        self.hotspot_inactive: bool = False
        self.hotspot_attempts: list[tuple[str, str | None]] = []
        self.forget_attempts: list[str] = []

    def get_status(self) -> WiFiStatus:
        return self.status

    def scan(self) -> list[WiFiNetwork]:
        return list(self.networks)

    def connect(self, ssid: str, password: str | None) -> WiFiStatus:
        self.connect_attempts.append((ssid, password))
        if ssid == "Guest":
            # Simulate an authentication issue that leaves us disconnected.
            self.status = WiFiStatus(
                connected=False,
                ssid=None,
                signal=None,
                ip_address=None,
                mode="station",
                hotspot_active=False,
                profile=None,
                detail="Authentication failed",
            )
        else:
            self.status = WiFiStatus(
                connected=True,
                ssid=ssid,
                signal=60,
                ip_address="192.168.1.20",
                mode="station",
                hotspot_active=False,
                profile=ssid,
                detail=f"Connected to {ssid}",
            )
        return self.status

    def activate_profile(self, profile: str) -> WiFiStatus:
        # Record the rollback attempt.
        self.connect_attempts.append((profile, None))
        self.status = WiFiStatus(
            connected=True,
            ssid="Home",
            signal=80,
            ip_address="192.168.1.10",
            mode="station",
            hotspot_active=False,
            profile="Home",
            detail="Restored previous connection",
        )
        return self.status

    def start_hotspot(self, ssid: str, password: str | None) -> WiFiStatus:
        self.hotspot_attempts.append((ssid, password))
        if self.hotspot_error:
            raise WiFiError(self.hotspot_error)
        if self.hotspot_inactive:
            self.status = WiFiStatus(
                connected=True,
                ssid=ssid or "RevCam",
                signal=None,
                ip_address="192.168.4.1",
                mode="station",
                hotspot_active=False,
                profile="rev-hotspot",
                detail="Activating hotspot…",
            )
            return self.status
        self.status = WiFiStatus(
            connected=True,
            ssid=ssid or "RevCam",
            signal=None,
            ip_address="192.168.4.1",
            mode="access-point",
            hotspot_active=True,
            profile="rev-hotspot",
            detail="Hotspot enabled",
        )
        return self.status

    def stop_hotspot(self, profile: str | None) -> WiFiStatus:
        self.status = WiFiStatus(
            connected=False,
            ssid=None,
            signal=None,
            ip_address=None,
            mode="station",
            hotspot_active=False,
            profile=None,
            detail="Hotspot disabled",
        )
        return self.status

    def forget(self, profile_or_ssid: str) -> None:
        identifier = (profile_or_ssid or "").strip()
        self.forget_attempts.append(identifier)
        self.networks = [network for network in self.networks if network.ssid != identifier]
        if self.status.profile == identifier:
            self.status = WiFiStatus(
                connected=False,
                ssid=None,
                signal=None,
                ip_address=None,
                mode="station",
                hotspot_active=False,
                profile=None,
                detail=f"Removed saved network {identifier}.",
            )
        else:
            self.status.detail = f"Removed saved network {identifier}."


class FakeMDNSAdvertiser:
    def __init__(self) -> None:
        self.calls: list[str | None] = []
        self.closed = False

    def advertise(self, ip_address: str | None) -> None:
        self.calls.append(ip_address)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def wifi_backend() -> FakeWiFiBackend:
    return FakeWiFiBackend()


@pytest.fixture
def mdns_advertiser() -> FakeMDNSAdvertiser:
    return FakeMDNSAdvertiser()


@pytest.fixture
def client(
    tmp_path: Path,
    wifi_backend: FakeWiFiBackend,
    mdns_advertiser: FakeMDNSAdvertiser,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    class _StubBatteryMonitor:
        def read(self):  # pragma: no cover - used indirectly
            return SimpleNamespace(to_dict=lambda: {"available": False})

    class _StubSupervisor:
        def start(self) -> None:  # pragma: no cover - used indirectly
            return None

        async def aclose(self) -> None:  # pragma: no cover - used indirectly
            return None

    credentials = WiFiCredentialStore(tmp_path / "wifi_credentials.json")
    manager = WiFiManager(
        backend=wifi_backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        mdns_advertiser=mdns_advertiser,
        credential_store=credentials,
    )
    monkeypatch.setattr("rev_cam.app.BatteryMonitor", lambda *args, **kwargs: _StubBatteryMonitor())
    monkeypatch.setattr("rev_cam.app.BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr("rev_cam.app.create_battery_overlay", lambda *args, **kwargs: (lambda frame: frame))
    app = create_app(tmp_path / "config.json", wifi_manager=manager)
    with TestClient(app) as test_client:
        test_client.backend = wifi_backend
        test_client.mdns = mdns_advertiser
        yield test_client


def test_wifi_status_endpoint(client: TestClient) -> None:
    response = client.get("/api/wifi/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert payload["ssid"] == "Home"
    assert payload["hotspot_active"] is False
    assert payload["hotspot_password"] == DEFAULT_HOTSPOT_PASSWORD


def test_wifi_scan_endpoint(client: TestClient) -> None:
    response = client.get("/api/wifi/networks")
    assert response.status_code == 200
    payload = response.json()
    assert "networks" in payload
    assert any(network.get("active") for network in payload["networks"])
    assert all("stored_credentials" in network for network in payload["networks"])


def test_wifi_scan_marks_stored_credentials(tmp_path: Path) -> None:
    backend = FakeWiFiBackend()
    credentials_path = tmp_path / "wifi_credentials.json"
    store = WiFiCredentialStore(credentials_path)
    store.set_network_password("Home", "homepass")
    manager = WiFiManager(
        backend=backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        credential_store=store,
    )

    networks = manager.scan_networks()

    assert any(network.ssid == "Home" and network.stored_credentials for network in networks)


def test_system_log_endpoint_includes_startup(client: TestClient) -> None:
    response = client.get("/api/logs")
    assert response.status_code == 200
    payload = response.json()
    entries = payload.get("entries", [])
    assert entries
    events = {entry.get("event") for entry in entries}
    assert "startup" in events or "startup_complete" in events
    assert any(entry.get("category") == "system" for entry in entries)


def test_wifi_connect_development_mode_rolls_back(client: TestClient) -> None:
    response = client.post(
        "/api/wifi/connect",
        json={"ssid": "Guest", "development_mode": True, "rollback_seconds": 0.05},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "restored" in (payload.get("detail") or "").lower()
    backend = getattr(client, "backend", None)
    assert backend is not None
    assert backend.connect_attempts[0][0] == "Guest"
    assert backend.connect_attempts[1][0] == "Home"


def test_wifi_connect_manual_rollback_without_dev_mode(client: TestClient) -> None:
    response = client.post(
        "/api/wifi/connect",
        json={"ssid": "Guest", "rollback_seconds": 0.05},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "restored" in (payload.get("detail") or "").lower()
    backend = getattr(client, "backend", None)
    assert backend is not None
    assert backend.connect_attempts[0][0] == "Guest"
    assert backend.connect_attempts[1][0] == "Home"


def test_wifi_connect_switch_prompts_network_change_notice(client: TestClient) -> None:
    backend = getattr(client, "backend", None)
    assert backend is not None
    backend.status = WiFiStatus(
        connected=True,
        ssid="Home",
        signal=70,
        ip_address="192.168.1.10",
        mode="station",
        hotspot_active=False,
        profile="Home",
        detail="Connected to Home",
    )

    response = client.post("/api/wifi/connect", json={"ssid": "Cafe"})
    assert response.status_code == 200
    payload = response.json()
    detail = (payload.get("detail") or "").lower()
    assert "reconnect" in detail
    assert "cafe" in detail


def test_wifi_connect_same_network_omits_switch_notice(client: TestClient) -> None:
    backend = getattr(client, "backend", None)
    assert backend is not None
    backend.status = WiFiStatus(
        connected=True,
        ssid="Home",
        signal=70,
        ip_address="192.168.1.10",
        mode="station",
        hotspot_active=False,
        profile="Home",
        detail="Connected to Home",
    )

    response = client.post("/api/wifi/connect", json={"ssid": "Home"})
    assert response.status_code == 200
    payload = response.json()
    detail = (payload.get("detail") or "").lower()
    assert "reconnect" not in detail


def test_wifi_hotspot_toggle(client: TestClient) -> None:
    enable = client.post(
        "/api/wifi/hotspot",
        json={"enabled": True, "ssid": "RevCam-AP", "password": "secret123"},
    )
    assert enable.status_code == 200
    assert enable.json()["hotspot_active"] is True
    advertiser = getattr(client, "mdns", None)
    assert advertiser is not None
    assert advertiser.calls
    assert advertiser.calls[-1] == "192.168.4.1"

    disable = client.post("/api/wifi/hotspot", json={"enabled": False})
    assert disable.status_code == 200
    assert disable.json()["hotspot_active"] is False
    assert advertiser.calls[-1] is None


def test_wifi_hotspot_password_persisted_in_status(client: TestClient) -> None:
    enable = client.post(
        "/api/wifi/hotspot",
        json={"enabled": True, "password": "secret123"},
    )
    assert enable.status_code == 200
    payload = enable.json()
    assert payload["hotspot_password"] == "secret123"

    status = client.get("/api/wifi/status")
    assert status.status_code == 200
    assert status.json()["hotspot_password"] == "secret123"

    disable = client.post("/api/wifi/hotspot", json={"enabled": False})
    assert disable.status_code == 200
    assert disable.json()["hotspot_password"] == "secret123"

    open_hotspot = client.post("/api/wifi/hotspot", json={"enabled": True})
    assert open_hotspot.status_code == 200
    assert open_hotspot.json()["hotspot_password"] == DEFAULT_HOTSPOT_PASSWORD

    refreshed = client.get("/api/wifi/status")
    assert refreshed.status_code == 200
    assert refreshed.json()["hotspot_password"] == DEFAULT_HOTSPOT_PASSWORD


def test_wifi_hotspot_password_persisted_across_restart(
    tmp_path: Path,
    wifi_backend: FakeWiFiBackend,
    mdns_advertiser: FakeMDNSAdvertiser,
) -> None:
    credentials_path = tmp_path / "wifi_credentials.json"
    manager = WiFiManager(
        backend=wifi_backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        mdns_advertiser=mdns_advertiser,
        credential_store=WiFiCredentialStore(credentials_path),
    )
    enabled = manager.enable_hotspot("RevCam", "secret123")
    assert enabled.hotspot_password == "secret123"

    restarted = WiFiManager(
        backend=wifi_backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        mdns_advertiser=mdns_advertiser,
        credential_store=WiFiCredentialStore(credentials_path),
    )
    status = restarted.get_status()
    assert status.hotspot_password == "secret123"


def test_wifi_hotspot_development_mode_rolls_back(client: TestClient) -> None:
    backend = getattr(client, "backend", None)
    assert backend is not None
    backend.hotspot_inactive = True
    response = client.post(
        "/api/wifi/hotspot",
        json={
            "enabled": True,
            "ssid": "RevCam-AP",
            "development_mode": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["hotspot_active"] is False
    assert "restored" in (payload.get("detail") or "").lower()
    assert backend.connect_attempts[-1][0] == "Home"


def test_wifi_hotspot_manual_rollback_without_dev_mode(client: TestClient) -> None:
    backend = getattr(client, "backend", None)
    assert backend is not None
    backend.hotspot_inactive = True
    response = client.post(
        "/api/wifi/hotspot",
        json={
            "enabled": True,
            "ssid": "RevCam-AP",
            "rollback_seconds": 0.05,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["hotspot_active"] is False
    assert "restored" in (payload.get("detail") or "").lower()
    assert backend.connect_attempts[-1][0] == "Home"


def test_wifi_hotspot_defaults_to_revcam_without_password(client: TestClient) -> None:
    backend = getattr(client, "backend", None)
    assert backend is not None
    response = client.post("/api/wifi/hotspot", json={"enabled": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ssid"] == "RevCam"
    assert backend.hotspot_attempts
    hotspot_ssid, hotspot_password = backend.hotspot_attempts[-1]
    assert hotspot_ssid == "RevCam"
    assert hotspot_password == DEFAULT_HOTSPOT_PASSWORD


def test_wifi_hotspot_permission_error(client: TestClient) -> None:
    backend = getattr(client, "backend", None)
    assert backend is not None
    backend.hotspot_error = (
        "Error: Failed to setup a Wi-Fi hotspot: Not authorized to control networking."
    )
    response = client.post(
        "/api/wifi/hotspot",
        json={"enabled": True, "ssid": "RevCam-AP"},
    )
    assert response.status_code == 400
    payload = response.json()
    assert "not authorized" in payload["detail"].lower()


def test_wifi_forget_endpoint_removes_network(client: TestClient) -> None:
    backend = getattr(client, "backend", None)
    assert backend is not None
    assert any(network.ssid == "Home" for network in backend.networks)

    response = client.post("/api/wifi/forget", json={"identifier": "Home"})
    assert response.status_code == 200
    payload = response.json()
    assert "removed saved network" in (payload.get("detail") or "").lower()

    assert backend.forget_attempts[-1] == "Home"

    refreshed = client.get("/api/wifi/networks")
    assert refreshed.status_code == 200
    networks = refreshed.json().get("networks", [])
    assert all(network.get("ssid") != "Home" for network in networks)


def test_wifi_log_records_connection_and_hotspot_events(client: TestClient) -> None:
    connect = client.post(
        "/api/wifi/connect",
        json={"ssid": "Guest", "development_mode": True, "rollback_seconds": 0.05},
    )
    assert connect.status_code == 200

    enable = client.post(
        "/api/wifi/hotspot",
        json={"enabled": True, "ssid": "RevCam-AP", "password": "secret123"},
    )
    assert enable.status_code == 200

    disable = client.post("/api/wifi/hotspot", json={"enabled": False})
    assert disable.status_code == 200

    log_response = client.get("/api/wifi/log")
    assert log_response.status_code == 200
    entries = log_response.json().get("entries", [])
    assert entries
    assert all(entry.get("category") == "network" for entry in entries)
    events = {entry.get("event") for entry in entries}
    assert "connect_attempt" in events
    assert "connect_rollback" in events or "connect_success" in events
    assert "hotspot_enable_attempt" in events
    assert "hotspot_disabled" in events

    system_response = client.get("/api/logs")
    assert system_response.status_code == 200
    system_entries = system_response.json().get("entries", [])
    system_events = {entry.get("event") for entry in system_entries}
    assert "connect_attempt" in system_events
    assert "hotspot_enable_attempt" in system_events


def test_wifi_network_password_reused_from_store(tmp_path: Path) -> None:
    credentials_path = tmp_path / "wifi_credentials.json"
    backend = FakeWiFiBackend()
    manager = WiFiManager(
        backend=backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        credential_store=WiFiCredentialStore(credentials_path),
    )
    manager.connect("Cafe", "beanjuice123")
    assert backend.connect_attempts[-1] == ("Cafe", "beanjuice123")

    new_backend = FakeWiFiBackend()
    reloaded = WiFiManager(
        backend=new_backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        credential_store=WiFiCredentialStore(credentials_path),
    )
    reloaded.connect("Cafe", None)
    assert new_backend.connect_attempts[-1] == ("Cafe", "beanjuice123")


def test_auto_connect_prefers_strongest_known_network(
    tmp_path: Path,
    wifi_backend: FakeWiFiBackend,
    mdns_advertiser: FakeMDNSAdvertiser,
) -> None:
    wifi_backend.status = WiFiStatus(
        connected=False,
        ssid=None,
        signal=None,
        ip_address=None,
        mode="station",
        hotspot_active=False,
        profile=None,
        detail="Disconnected",
    )
    wifi_backend.networks = [
        WiFiNetwork(
            ssid="Home",
            signal=45,
            security="WPA2",
            frequency=2412.0,
            channel=1,
            known=True,
            active=False,
        ),
        WiFiNetwork(
            ssid="Cafe",
            signal=82,
            security="WPA2",
            frequency=2462.0,
            channel=11,
            known=True,
            active=False,
        ),
    ]
    credentials = WiFiCredentialStore(tmp_path / "wifi_credentials.json")
    credentials.set_network_password("Home", "homepass")
    credentials.set_network_password("Cafe", "cafepass")
    manager = WiFiManager(
        backend=wifi_backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        mdns_advertiser=mdns_advertiser,
        credential_store=credentials,
    )

    status = manager.auto_connect_known_networks()

    assert wifi_backend.connect_attempts
    assert wifi_backend.connect_attempts[0][0] == "Cafe"
    assert status.connected is True
    assert status.ssid == "Cafe"


def test_auto_connect_falls_back_to_hotspot(
    tmp_path: Path,
    wifi_backend: FakeWiFiBackend,
    mdns_advertiser: FakeMDNSAdvertiser,
) -> None:
    wifi_backend.status = WiFiStatus(
        connected=False,
        ssid=None,
        signal=None,
        ip_address=None,
        mode="station",
        hotspot_active=False,
        profile=None,
        detail="Disconnected",
    )
    wifi_backend.networks = [
        WiFiNetwork(
            ssid="Guest",
            signal=60,
            security="WPA2",
            frequency=2417.0,
            channel=2,
            known=True,
            active=False,
        )
    ]
    manager = WiFiManager(
        backend=wifi_backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        mdns_advertiser=mdns_advertiser,
        credential_store=WiFiCredentialStore(tmp_path / "wifi_credentials.json"),
    )

    status = manager.auto_connect_known_networks()

    assert wifi_backend.connect_attempts
    assert wifi_backend.connect_attempts[0][0] == "Guest"
    assert wifi_backend.hotspot_attempts
    assert status.hotspot_active is True


def test_auto_connect_enables_hotspot_when_connected_to_unknown_network(
    tmp_path: Path,
    wifi_backend: FakeWiFiBackend,
    mdns_advertiser: FakeMDNSAdvertiser,
) -> None:
    wifi_backend.status = WiFiStatus(
        connected=True,
        ssid="Visitor",
        signal=45,
        ip_address="192.168.50.12",
        mode="station",
        hotspot_active=False,
        profile="Visitor",
        detail="Connected to Visitor",
    )
    wifi_backend.networks = []
    manager = WiFiManager(
        backend=wifi_backend,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        mdns_advertiser=mdns_advertiser,
        credential_store=WiFiCredentialStore(tmp_path / "wifi_credentials.json"),
    )

    status = manager.auto_connect_known_networks()

    assert wifi_backend.hotspot_attempts
    assert status.hotspot_active is True
