import time
from pathlib import Path

import pytest

from rev_cam.wifi import WiFiCredentialStore, WiFiManager, WiFiNetwork, WiFiStatus


class WatchdogBackend:
    def __init__(self) -> None:
        self.status = WiFiStatus(
            connected=False,
            ssid=None,
            signal=None,
            ip_address=None,
            mode="station",
            hotspot_active=False,
            profile=None,
            detail="Disconnected",
        )
        self.networks: list[WiFiNetwork] = []
        self.connect_calls: list[tuple[str, str | None]] = []
        self.hotspot_calls: list[tuple[str, str | None]] = []

    def get_status(self) -> WiFiStatus:
        return self.status

    def scan(self) -> list[WiFiNetwork]:
        return list(self.networks)

    def connect(self, ssid: str, password: str | None) -> WiFiStatus:
        self.connect_calls.append((ssid, password))
        self.status = WiFiStatus(
            connected=False,
            ssid=None,
            signal=None,
            ip_address=None,
            mode="station",
            hotspot_active=False,
            profile=None,
            detail=f"Failed to join {ssid}",
        )
        return self.status

    def activate_profile(self, profile: str) -> WiFiStatus:
        return self.status

    def start_hotspot(self, ssid: str, password: str | None) -> WiFiStatus:
        self.hotspot_calls.append((ssid, password))
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

    def forget(self, profile_or_ssid: str) -> None:  # pragma: no cover - not used
        return None


@pytest.fixture
def watchdog_manager(tmp_path: Path) -> tuple[WiFiManager, WatchdogBackend, WiFiCredentialStore]:
    backend = WatchdogBackend()
    credentials = WiFiCredentialStore((tmp_path / "watchdog").resolve() / "wifi.json")
    manager = WiFiManager(
        backend=backend,
        credential_store=credentials,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        watchdog_boot_delay=0.02,
        watchdog_interval=0.05,
        watchdog_retry_delay=0.01,
    )
    return manager, backend, credentials


def test_watchdog_enables_hotspot_after_boot_delay(
    watchdog_manager: tuple[WiFiManager, WatchdogBackend, WiFiCredentialStore]
) -> None:
    manager, backend, _ = watchdog_manager
    manager.start_hotspot_watchdog()
    try:
        time.sleep(0.08)
        assert backend.hotspot_calls
    finally:
        manager.close()


def test_watchdog_skips_when_known_network_connected(
    watchdog_manager: tuple[WiFiManager, WatchdogBackend, WiFiCredentialStore]
) -> None:
    manager, backend, credentials = watchdog_manager
    credentials.set_network_password("Home", "secret123")
    backend.status = WiFiStatus(
        connected=True,
        ssid="Home",
        signal=72,
        ip_address="192.168.1.2",
        mode="station",
        hotspot_active=False,
        profile="Home",
        detail="Connected",
    )
    manager.start_hotspot_watchdog()
    try:
        time.sleep(0.06)
        assert backend.hotspot_calls == []
        assert backend.connect_calls == []
    finally:
        manager.close()


def test_watchdog_retries_before_enabling_hotspot(
    watchdog_manager: tuple[WiFiManager, WatchdogBackend, WiFiCredentialStore]
) -> None:
    manager, backend, credentials = watchdog_manager
    credentials.set_network_password("Home", "secret123")
    backend.networks = [
        WiFiNetwork(
            ssid="Home",
            signal=60,
            security="WPA2",
            frequency=2412.0,
            channel=1,
            known=True,
            active=False,
        )
    ]
    manager.start_hotspot_watchdog()
    try:
        time.sleep(0.08)
        assert len(backend.connect_calls) >= 2
        assert backend.hotspot_calls
    finally:
        manager.close()
