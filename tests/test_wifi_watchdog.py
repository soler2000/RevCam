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
        self.hotspot_error_details: list[dict[str, object | None]] = []
        self.hotspot_inactive_attempts: int = 0
        self.hotspot_diagnostics_payloads: list[dict[str, object | None]] = []
        self._diagnostic_index = 0

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
            exc = WiFiError(message)
            if self.hotspot_error_details:
                setattr(exc, "details", self.hotspot_error_details.pop(0))
            raise exc
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

    def hotspot_diagnostics(self) -> dict[str, object | None] | None:
        if self.hotspot_diagnostics_payloads:
            index = min(self._diagnostic_index, len(self.hotspot_diagnostics_payloads) - 1)
            self._diagnostic_index += 1
            return self.hotspot_diagnostics_payloads[index]
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
    backend.hotspot_error_details = [
        {
            "connection": "RevCam Hotspot",
            "attempts": [
                {
                    "stage": "initial",
                    "steps": [
                        {"action": "remove_setting", "result": "error"},
                        {"property": "802-11-wireless-security.psk", "action": "remove", "result": "error"},
                    ],
                }
            ],
        },
        {
            "connection": "RevCam Hotspot",
            "attempts": [
                {
                    "stage": "initial",
                    "steps": [
                        {"action": "remove_setting", "result": "error"},
                        {"property": "802-11-wireless-security.key-mgmt", "action": "set", "result": "error"},
                    ],
                }
            ],
        },
    ]
    backend.hotspot_diagnostics_payloads = [
        {"state": "first-attempt"},
        {"state": "second-attempt"},
    ]
    manager.start_hotspot_watchdog()
    try:
        deadline = time.time() + 1.0
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
        error_entries = sorted(
            (
                entry
                for entry in entries
                if entry["event"] == "hotspot_watchdog_hotspot_error"
            ),
            key=lambda entry: entry.get("metadata", {}).get("attempt", 0),
        )
        assert len(error_entries) >= 2
        diag_states: set[str] = set()
        for index, entry in enumerate(error_entries[:2], start=1):
            metadata = entry.get("metadata")
            assert metadata is not None
            assert metadata.get("attempt") == index
            assert metadata.get("trigger") == "watchdog"
            assert "error_details" in metadata
            assert "error_details_json" in metadata
            assert metadata["error_details"]["connection"] == "RevCam Hotspot"
            diag_state = metadata.get("diagnostics", {}).get("state")
            assert diag_state in {"first-attempt", "second-attempt"}
            diag_states.add(diag_state)
            summary = metadata.get("diagnostics_summary")
            assert isinstance(summary, str) and summary
            assert "initial" in summary
            requires_secret_step = False
            error_details = metadata.get("error_details")
            attempts_payload = []
            if isinstance(error_details, dict):
                attempts_payload = error_details.get("attempts", []) or []
            for attempt in attempts_payload:
                if not isinstance(attempt, dict):
                    continue
                for step in attempt.get("steps", []) or []:
                    if not isinstance(step, dict):
                        continue
                    if step.get("error") == "hotspot security still expects secrets":
                        requires_secret_step = True
                        break
                if requires_secret_step:
                    break
            if requires_secret_step:
                assert "hotspot security still expects secrets" in summary
            assert "diagnostics_json" in metadata
        assert diag_states
        assert diag_states.issubset({"first-attempt", "second-attempt"})
        final_entry = next(
            entry for entry in entries if entry["event"] == "hotspot_watchdog_hotspot_failed"
        )
        final_metadata = final_entry.get("metadata")
        assert final_metadata is not None
        attempts = final_metadata.get("attempts")
        assert isinstance(attempts, list) and len(attempts) >= 2
        assert attempts[0].get("attempt") == 1
        assert attempts[1].get("attempt") == 2
        assert "diagnostics_json" in final_metadata
        assert "error_details_json" in final_metadata
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


def test_enable_hotspot_error_metadata(tmp_path: Path) -> None:
    backend = WatchdogBackend()
    backend.hotspot_error_messages = ["failed to configure"]
    backend.hotspot_error_details = [
        {
            "connection": "RevCam Hotspot",
            "attempts": [
                {
                    "stage": "initial",
                    "steps": [
                        {"action": "remove_setting", "result": "error"},
                        {"property": "802-11-wireless-security.psk", "action": "set", "result": "error"},
                    ],
                }
            ],
        }
    ]
    backend.hotspot_diagnostics_payloads = [{"state": "enable"}]
    base_dir = (tmp_path / "enable").resolve()
    credentials = WiFiCredentialStore(base_dir / "wifi.json")
    manager = WiFiManager(
        backend=backend,
        credential_store=credentials,
        rollback_timeout=0.05,
        poll_interval=0.005,
        hotspot_rollback_timeout=0.05,
        watchdog_boot_delay=0.02,
        watchdog_interval=0.05,
        watchdog_retry_delay=0.01,
        log_path=base_dir / "wifi_log.jsonl",
    )
    with pytest.raises(WiFiError):
        manager.enable_hotspot()
    try:
        entries = manager.get_connection_log()
        error_entry = next(
            entry for entry in entries if entry["event"] == "hotspot_enable_error"
        )
        metadata = error_entry.get("metadata")
        assert metadata is not None
        assert metadata.get("error_details", {}).get("connection") == "RevCam Hotspot"
        assert metadata.get("diagnostics", {}).get("state") == "enable"
    finally:
        manager.close()
