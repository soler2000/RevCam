from collections import Counter
import time
from pathlib import Path

import pytest

from rev_cam.wifi import WiFiCredentialStore, WiFiError, WiFiManager, WiFiNetwork, WiFiStatus


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
        self.hotspot_error_messages: list[str] = []
        self.hotspot_inactive_attempts: int = 0

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
        if self.hotspot_error_messages:
            message = self.hotspot_error_messages.pop(0)
            raise WiFiError(message)
        if self.hotspot_inactive_attempts > 0:
            self.hotspot_inactive_attempts -= 1
            self.status = WiFiStatus(
                connected=True,
                ssid=ssid or "RevCam",
                signal=None,
                ip_address="192.168.4.1",
                mode="station",
                hotspot_active=False,
                profile="rev-hotspot",
                detail="Hotspot pending",
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
        log_path=(tmp_path / "watchdog" / "wifi_log.jsonl"),
    )
    return manager, backend, credentials


def test_watchdog_enables_hotspot_after_boot_delay(
    watchdog_manager: tuple[WiFiManager, WatchdogBackend, WiFiCredentialStore]
) -> None:
    manager, backend, _ = watchdog_manager
    manager.start_hotspot_watchdog()
    try:
        time.sleep(0.2)
    finally:
        manager.close()
    assert backend.hotspot_calls


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
    finally:
        manager.close()
    assert backend.hotspot_calls == []
    assert backend.connect_calls == []


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
        time.sleep(0.5)
        assert len(backend.connect_calls) >= 2
        assert backend.hotspot_calls
    finally:
        manager.close()


def test_watchdog_enables_hotspot_when_connection_drops(
    watchdog_manager: tuple[WiFiManager, WatchdogBackend, WiFiCredentialStore]
) -> None:
    manager, backend, credentials = watchdog_manager
    credentials.set_network_password("Home", "secret123")
    backend.status = WiFiStatus(
        connected=True,
        ssid="Home",
        signal=70,
        ip_address="192.168.1.2",
        mode="station",
        hotspot_active=False,
        profile="Home",
        detail="Connected",
    )
    manager.start_hotspot_watchdog()
    try:
        time.sleep(0.03)
        backend.status = WiFiStatus(
            connected=False,
            ssid=None,
            signal=None,
            ip_address=None,
            mode="station",
            hotspot_active=False,
            profile=None,
            detail="Connection lost",
        )
        backend.networks = []
        time.sleep(0.4)
        assert backend.hotspot_calls
    finally:
        manager.close()


def test_watchdog_retries_hotspot_when_inactive(
    watchdog_manager: tuple[WiFiManager, WatchdogBackend, WiFiCredentialStore]
) -> None:
    manager, backend, _ = watchdog_manager
    backend.hotspot_inactive_attempts = 1
    manager.start_hotspot_watchdog()
    try:
        time.sleep(1.0)
    finally:
        manager.close()
    entries = manager.get_connection_log()
    assert any(entry["event"] == "hotspot_watchdog_hotspot_enabled" for entry in entries)
    assert any(entry["event"] == "hotspot_watchdog_hotspot_inactive" for entry in entries)


def test_watchdog_reports_failure_when_hotspot_errors(
    watchdog_manager: tuple[WiFiManager, WatchdogBackend, WiFiCredentialStore]
) -> None:
    manager, backend, _ = watchdog_manager
    backend.hotspot_error_messages = ["busy", "still busy"]
    manager.start_hotspot_watchdog()
    try:
        deadline = time.time() + 0.5
        failure_logged = False
        error_count = 0
        while time.time() < deadline and not failure_logged:
            entries = manager.get_connection_log()
            counts = Counter(entry["event"] for entry in entries)
            error_count = counts.get("hotspot_watchdog_hotspot_error", 0)
            failure_logged = any(
                entry["event"] == "hotspot_watchdog_hotspot_failed" for entry in entries
            )
            if failure_logged:
                break
            time.sleep(0.02)
        assert error_count >= 2
        assert failure_logged
        entries = manager.get_connection_log()
        assert not any(entry["event"] == "hotspot_watchdog_hotspot_enabled" for entry in entries)
    finally:
        manager.close()


def test_watchdog_persistent_log(tmp_path: Path) -> None:
    backend = WatchdogBackend()
    base_dir = (tmp_path / "watchdog").resolve()
    credentials = WiFiCredentialStore(base_dir / "wifi.json")
    log_path = base_dir / "wifi_log.jsonl"
    manager = WiFiManager(
        backend=backend,
        credential_store=credentials,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        watchdog_boot_delay=0.02,
        watchdog_interval=0.05,
        watchdog_retry_delay=0.01,
        log_path=log_path,
    )
    manager._record_log("test_event", "Stored before restart.")
    manager.close()

    restarted = WiFiManager(
        backend=backend,
        credential_store=credentials,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        watchdog_boot_delay=0.02,
        watchdog_interval=0.05,
        watchdog_retry_delay=0.01,
        log_path=log_path,
    )
    try:
        entries = restarted.get_connection_log()
        assert any(entry["event"] == "test_event" for entry in entries)
    finally:
        restarted.close()
