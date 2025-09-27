"""Wi-Fi management utilities with safe fallbacks for RevCam."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
import math
import subprocess
import threading
import time
from typing import Deque, Iterable, Sequence

from .mdns import MDNSAdvertiser


class WiFiError(RuntimeError):
    """Raised when Wi-Fi operations fail."""


def _channel_from_frequency(freq_mhz: float | None) -> int | None:
    """Best-effort conversion from MHz to Wi-Fi channel numbers."""

    if freq_mhz is None or freq_mhz <= 0:
        return None
    # IEEE 802.11 2.4 GHz channels use a 5 MHz spacing starting at 2412 MHz.
    if 2400 <= freq_mhz <= 2500:
        channel = int(round((freq_mhz - 2407) / 5))
        if 1 <= channel <= 14:
            return channel
        return None
    # 5 GHz allocations start around 5035 MHz depending on the regulatory domain.
    if 4900 <= freq_mhz <= 5900:
        channel = int(round((freq_mhz - 5000) / 5))
        if channel > 0:
            return channel
        return None
    return None


@dataclass(slots=True)
class WiFiNetwork:
    """Represents a Wi-Fi network discovered during a scan."""

    ssid: str
    signal: int | None = None
    security: str | None = None
    frequency: float | None = None
    channel: int | None = None
    known: bool = False
    active: bool = False
    hidden: bool = False

    def to_dict(self) -> dict[str, object | None]:
        return {
            "ssid": self.ssid,
            "signal": self.signal,
            "security": self.security,
            "frequency": self.frequency,
            "channel": self.channel,
            "known": self.known,
            "active": self.active,
            "hidden": self.hidden,
        }


@dataclass(slots=True)
class WiFiStatus:
    """Represents the current Wi-Fi connection state."""

    connected: bool
    ssid: str | None
    signal: int | None
    ip_address: str | None
    mode: str = "unknown"
    hotspot_active: bool = False
    profile: str | None = None
    error: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object | None]:
        return {
            "connected": self.connected,
            "ssid": self.ssid,
            "signal": self.signal,
            "ip_address": self.ip_address,
            "mode": self.mode,
            "hotspot_active": self.hotspot_active,
            "profile": self.profile,
            "error": self.error,
            "detail": self.detail,
        }


@dataclass(slots=True)
class WiFiLogEntry:
    """Represents a Wi-Fi operation event for troubleshooting."""

    timestamp: float
    event: str
    message: str
    status: dict[str, object | None] | None = None
    metadata: dict[str, object | None] | None = None

    def to_dict(self) -> dict[str, object | None]:
        payload: dict[str, object | None] = {
            "timestamp": self.timestamp,
            "event": self.event,
            "message": self.message,
        }
        if self.status is not None:
            payload["status"] = self.status
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class WiFiBackend:
    """Abstract interface for Wi-Fi operations."""

    def get_status(self) -> WiFiStatus:  # pragma: no cover - interface only
        raise NotImplementedError

    def scan(self) -> Sequence[WiFiNetwork]:  # pragma: no cover - interface only
        raise NotImplementedError

    def connect(self, ssid: str, password: str | None) -> WiFiStatus:  # pragma: no cover - interface only
        raise NotImplementedError

    def activate_profile(self, profile: str) -> WiFiStatus:  # pragma: no cover - interface only
        raise NotImplementedError

    def start_hotspot(self, ssid: str, password: str | None) -> WiFiStatus:  # pragma: no cover - interface only
        raise NotImplementedError

    def stop_hotspot(self, profile: str | None) -> WiFiStatus:  # pragma: no cover - interface only
        raise NotImplementedError

    def forget(self, profile_or_ssid: str) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class NMCLIBackend(WiFiBackend):
    """Interact with NetworkManager via nmcli commands."""

    def __init__(self, interface: str | None = None, *, timeout: float = 15.0) -> None:
        self._preferred_interface = interface
        self._timeout = timeout
        self._detected_interface: str | None = None
        self._last_hotspot_profile: str | None = None

    # ------------------------------- helpers -------------------------------
    def _run(self, args: Sequence[str]) -> str:
        try:
            completed = subprocess.run(
                list(args),
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except FileNotFoundError as exc:  # pragma: no cover - environment specific
            raise WiFiError("nmcli command unavailable") from exc
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - environment specific
            raise WiFiError("nmcli command timed out") from exc
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise WiFiError(error_output)
        return completed.stdout

    def _detect_interface(self) -> str:
        if self._preferred_interface:
            return self._preferred_interface
        if self._detected_interface:
            return self._detected_interface
        output = self._run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "device"])
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            device, dev_type, state = parts[:3]
            if dev_type.strip() == "wifi" and state.strip() != "unavailable":
                self._detected_interface = device.strip()
                return self._detected_interface
        raise WiFiError("No Wi-Fi interface detected")

    def _get_interface(self) -> str:
        return self._preferred_interface or self._detect_interface()

    def _parse_saved_profiles(self) -> set[str]:
        try:
            output = self._run(["nmcli", "-t", "-f", "NAME,TYPE", "connection", "show"])
        except WiFiError:
            return set()
        profiles: set[str] = set()
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            name, conn_type = parts[0].strip(), parts[1].strip()
            if conn_type == "802-11-wireless" and name:
                profiles.add(name)
        return profiles

    @staticmethod
    def _unescape_nmcli_field(value: str) -> str:
        """Best effort unescaping for nmcli's colon-delimited output."""

        if "\\" not in value:
            return value
        # nmcli escapes literal backslashes and colons. It does not use more
        # elaborate sequences, so a simple two-step replacement is sufficient.
        return value.replace("\\\\", "\\").replace("\\:", ":")

    def _scan_output(self, *, rescan: bool = False) -> Iterable[WiFiNetwork]:
        interface = self._get_interface()
        if rescan:
            try:
                self._run(["nmcli", "device", "wifi", "rescan", "ifname", interface])
            except WiFiError as exc:
                message = str(exc).strip()
                lowered = message.lower()
                if "not authorized" in lowered or "not authorised" in lowered:
                    raise WiFiError(
                        "Unable to rescan Wi-Fi networks: not authorized to control networking. "
                        "Ensure RevCam has permission to manage NetworkManager."
                    ) from exc
                # Some drivers refuse to rescan while already scanning or
                # operating as an access point. In those cases we continue with
                # the cached list to at least surface previously discovered
                # networks.
        args = [
            "nmcli",
            "-t",
            "-f",
            "IN-USE,SSID,SIGNAL,SECURITY,FREQ",
            "device",
            "wifi",
            "list",
            "ifname",
            interface,
        ]
        if rescan:
            args.extend(["--rescan", "yes"])
        output = self._run(args)
        saved = self._parse_saved_profiles()
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split(":")
            while len(parts) < 5:
                parts.append("")
            in_use, ssid_raw, signal_raw, security_raw, freq_raw = parts[:5]
            ssid_raw = self._unescape_nmcli_field(ssid_raw)
            ssid = ssid_raw.strip()
            hidden = not bool(ssid)
            if hidden:
                ssid = "(hidden network)"
            signal = None
            if signal_raw.strip():
                try:
                    signal = int(float(signal_raw.strip()))
                except ValueError:
                    signal = None
            frequency = None
            if freq_raw.strip():
                freq_text = freq_raw.strip().split()[0]
                try:
                    frequency = float(freq_text)
                except ValueError:
                    frequency = None
            security = security_raw.strip() or "unknown"
            network = WiFiNetwork(
                ssid=ssid,
                signal=signal,
                security=security,
                frequency=frequency,
                channel=_channel_from_frequency(frequency),
                known=ssid_raw.strip() in saved,
                active=in_use.strip() in {"*", "yes"},
                hidden=hidden,
            )
            yield network

    # ---------------------------- interface impl ---------------------------
    def get_status(self) -> WiFiStatus:
        interface = self._get_interface()
        output = self._run(
            [
                "nmcli",
                "-t",
                "-f",
                "GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS",
                "device",
                "show",
                interface,
            ]
        )
        details: dict[str, str] = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            details[key.strip()] = value.strip()
        state_text = details.get("GENERAL.STATE", "")
        connected = "connected" in state_text.lower() or state_text.startswith("100")
        connection_name = details.get("GENERAL.CONNECTION") or None
        ip_raw = details.get("IP4.ADDRESS[1]") or details.get("IP4.ADDRESS")
        ip_address = None
        if ip_raw:
            ip_address = ip_raw.split("/")[0].strip()
        signal = None
        ssid = connection_name
        for network in self._scan_output(rescan=False):
            if network.active:
                ssid = network.ssid if not network.hidden else connection_name
                signal = network.signal
                break
        mode = "unknown"
        hotspot_active = False
        if connection_name:
            try:
                mode_output = self._run(
                    [
                        "nmcli",
                        "-t",
                        "-f",
                        "802-11-wireless.mode",
                        "connection",
                        "show",
                        connection_name,
                    ]
                )
            except WiFiError:
                mode_output = ""
            for line in mode_output.splitlines():
                if line.startswith("802-11-wireless.mode:"):
                    mode_value = line.split(":", 1)[1].strip().lower()
                    if mode_value == "ap":
                        mode = "access-point"
                        hotspot_active = True
                    elif mode_value:
                        mode = "station"
                    break
        status = WiFiStatus(
            connected=connected,
            ssid=ssid,
            signal=signal,
            ip_address=ip_address,
            mode=mode,
            hotspot_active=hotspot_active,
            profile=connection_name,
        )
        if hotspot_active and connection_name:
            self._last_hotspot_profile = connection_name
        return status

    def scan(self) -> Sequence[WiFiNetwork]:
        return list(self._scan_output(rescan=True))

    def connect(self, ssid: str, password: str | None) -> WiFiStatus:
        """Connect to a Wi-Fi network, preferring saved profiles when available."""

        if not password:
            try:
                saved_profiles = self._parse_saved_profiles()
            except WiFiError:
                saved_profiles = set()
            if ssid in saved_profiles:
                return self.activate_profile(ssid)

        interface = self._get_interface()
        args = ["nmcli", "device", "wifi", "connect", ssid]
        if password:
            args.extend(["password", password])
        args.extend(["ifname", interface])
        self._run(args)
        return self.get_status()

    def activate_profile(self, profile: str) -> WiFiStatus:
        interface = self._get_interface()
        args = ["nmcli", "connection", "up", profile]
        args.extend(["ifname", interface])
        self._run(args)
        return self.get_status()

    def start_hotspot(self, ssid: str, password: str | None) -> WiFiStatus:
        interface = self._get_interface()
        connection_name = "RevCam Hotspot"
        args = [
            "nmcli",
            "device",
            "wifi",
            "hotspot",
            "ifname",
            interface,
            "con-name",
            connection_name,
            "ssid",
            ssid,
        ]
        if password:
            args.extend(["password", password])
        self._run(args)
        status = self.get_status()
        status.profile = connection_name
        status.mode = "access-point"
        status.hotspot_active = True
        self._last_hotspot_profile = connection_name
        return status

    def stop_hotspot(self, profile: str | None) -> WiFiStatus:
        connection_name = profile or self._last_hotspot_profile or "RevCam Hotspot"
        try:
            self._run(["nmcli", "connection", "down", connection_name])
        except WiFiError:
            # If the connection is already down, ignore and continue with status refresh.
            pass
        status = self.get_status()
        if not status.hotspot_active:
            self._last_hotspot_profile = None
        return status

    def forget(self, profile_or_ssid: str) -> None:
        try:
            self._run(["nmcli", "connection", "delete", profile_or_ssid])
        except WiFiError as exc:
            message = str(exc).strip()
            lowered = message.lower()
            if "not authorized" in lowered or "not authorised" in lowered:
                raise WiFiError(
                    "Unable to forget Wi-Fi network: not authorized to control networking. "
                    "Ensure RevCam has permission to manage NetworkManager."
                ) from exc
            raise
        if profile_or_ssid == self._last_hotspot_profile:
            self._last_hotspot_profile = None


class WiFiManager:
    """High-level orchestration with rollback for development mode."""

    def __init__(
        self,
        backend: WiFiBackend | None = None,
        *,
        rollback_timeout: float = 30.0,
        poll_interval: float = 1.0,
        hotspot_rollback_timeout: float | None = 120.0,
        mdns_advertiser: MDNSAdvertiser | None = None,
    ) -> None:
        self._backend = backend or NMCLIBackend()
        self._rollback_timeout = rollback_timeout
        self._poll_interval = max(0.1, poll_interval)
        self._hotspot_profile: str | None = None
        self._hotspot_rollback_timeout = (
            self._rollback_timeout
            if hotspot_rollback_timeout is None
            else max(0.0, hotspot_rollback_timeout)
        )
        self._mdns = mdns_advertiser
        if self._mdns is None:
            try:
                self._mdns = MDNSAdvertiser()
            except Exception as exc:  # pragma: no cover - environment specific
                logging.getLogger(__name__).warning(
                    "mDNS advertising disabled: %s", exc
                )
                self._mdns = None
        self._log: Deque[WiFiLogEntry] = deque(maxlen=200)
        self._log_lock = threading.Lock()
        try:
            initial_status = self._backend.get_status()
        except WiFiError:
            initial_status = None
        if initial_status:
            self._update_mdns(initial_status)

    # ------------------------------ operations -----------------------------
    def get_status(self) -> WiFiStatus:
        return self._fetch_status()

    def scan_networks(self) -> Sequence[WiFiNetwork]:
        return self._backend.scan()

    def forget_network(self, profile_or_ssid: str) -> WiFiStatus:
        if not isinstance(profile_or_ssid, str) or not profile_or_ssid.strip():
            raise WiFiError("Network identifier must be a non-empty string")
        identifier = profile_or_ssid.strip()
        self._backend.forget(identifier)
        if identifier == self._hotspot_profile:
            self._hotspot_profile = None
        status = self._fetch_status()
        return status

    def connect(
        self,
        ssid: str,
        password: str | None = None,
        *,
        development_mode: bool = False,
        rollback_timeout: float | None = None,
    ) -> WiFiStatus:
        if not isinstance(ssid, str) or not ssid.strip():
            raise WiFiError("SSID must be a non-empty string")
        cleaned_ssid = ssid.strip()
        if password is None:
            cleaned_password: str | None = None
        elif isinstance(password, str):
            cleaned_password = password if password.strip() else None
        else:
            raise WiFiError("Password must be a string or null")
        attempt_metadata: dict[str, object | None] = {
            "target": cleaned_ssid,
            "development_mode": development_mode,
        }
        if rollback_timeout is not None:
            attempt_metadata["requested_rollback"] = max(0.0, rollback_timeout)
        attempt_metadata["password_provided"] = cleaned_password is not None
        self._record_log(
            "connect_attempt",
            f"Attempting to connect to {cleaned_ssid}.",
            metadata=attempt_metadata,
        )
        previous_status = self._fetch_status()
        previous_profile = previous_status.profile if previous_status.connected else None
        try:
            status = self._backend.connect(cleaned_ssid, cleaned_password)
        except WiFiError as exc:
            message = str(exc).strip() or "Connection failed"
            self._record_log(
                "connect_error",
                f"Connection to {cleaned_ssid} failed: {message}.",
                metadata=attempt_metadata,
            )
            raise
        self._update_mdns(status)
        effective_timeout = None
        if development_mode:
            base_timeout = (
                self._rollback_timeout if rollback_timeout is None else max(0.0, rollback_timeout)
            )
            effective_timeout = base_timeout
        result_metadata = dict(attempt_metadata)
        if effective_timeout is not None:
            result_metadata["effective_rollback"] = effective_timeout
        monitor_required = False
        if status.connected and (status.ssid == cleaned_ssid or status.profile == cleaned_ssid):
            message = f"Connected to {status.ssid or cleaned_ssid}."
            self._record_log("connect_success", message, status=status, metadata=result_metadata)
            if not development_mode:
                return status
            timeout = effective_timeout
            if timeout is None or timeout <= 0:
                return status
            monitor_required = False
        else:
            detail = status.detail or "Connection did not report as active."
            message = f"Connection to {cleaned_ssid} pending: {detail}"
            self._record_log("connect_status", message, status=status, metadata=result_metadata)
            if not development_mode:
                return status
            timeout = (
                effective_timeout
                if effective_timeout is not None
                else self._rollback_timeout
                if rollback_timeout is None
                else max(0.0, rollback_timeout)
            )
            monitor_required = True
        if timeout <= 0 or not monitor_required:
            return status
        self._record_log(
            "connect_monitor",
            f"Monitoring connection to {cleaned_ssid} for up to {int(math.ceil(timeout))}s before rollback.",
            status=status,
            metadata=result_metadata,
        )
        if not previous_profile:
            detail = status.detail or ""
            status.detail = (
                f"{detail + ' ' if detail else ''}Development rollback skipped; previous network unknown."
            )
            self._record_log(
                "connect_no_previous_profile",
                status.detail,
                status=status,
                metadata=result_metadata,
            )
            return status
        deadline = time.monotonic() + timeout
        current = status
        while time.monotonic() < deadline:
            if current.connected and (current.ssid == cleaned_ssid or current.profile == cleaned_ssid):
                self._record_log(
                    "connect_confirmed",
                    f"Confirmed connection to {cleaned_ssid} before rollback deadline.",
                    status=current,
                    metadata=result_metadata,
                )
                return current
            time.sleep(self._poll_interval)
            current = self._fetch_status()
        restored = self._backend.activate_profile(previous_profile)
        self._update_mdns(restored)
        restored.detail = (
            f"Connection to {cleaned_ssid} did not establish within {int(math.ceil(timeout))}s; "
            f"restored {previous_status.ssid or previous_profile}."
        )
        rollback_metadata = dict(result_metadata)
        rollback_metadata["restored_profile"] = previous_profile
        self._record_log(
            "connect_rollback",
            restored.detail,
            status=restored,
            metadata=rollback_metadata,
        )
        return restored

    def enable_hotspot(
        self,
        ssid: str,
        password: str | None = None,
        *,
        development_mode: bool = False,
        rollback_timeout: float | None = None,
    ) -> WiFiStatus:
        if not isinstance(ssid, str) or not ssid.strip():
            raise WiFiError("Hotspot SSID must be provided")
        cleaned_ssid = ssid.strip()
        if password is not None and password.strip() and len(password.strip()) < 8:
            raise WiFiError("Hotspot password must be at least 8 characters")
        cleaned_password = password.strip() if isinstance(password, str) and password.strip() else None
        attempt_metadata: dict[str, object | None] = {
            "target": cleaned_ssid,
            "development_mode": development_mode,
            "password_provided": cleaned_password is not None,
        }
        if rollback_timeout is not None:
            attempt_metadata["requested_rollback"] = max(0.0, rollback_timeout)
        self._record_log(
            "hotspot_enable_attempt",
            f"Enabling hotspot {cleaned_ssid}.",
            metadata=attempt_metadata,
        )
        try:
            previous_status = self._fetch_status()
        except WiFiError:
            previous_status = None
            previous_profile = None
        else:
            previous_profile = previous_status.profile if previous_status.profile else None
        try:
            status = self._backend.start_hotspot(cleaned_ssid, cleaned_password)
        except WiFiError as exc:
            message = str(exc).strip()
            lower_message = message.lower()
            if "not authorized" in lower_message or "not authorised" in lower_message:
                raise WiFiError(
                    "Unable to enable hotspot: not authorized to control networking. "
                    "Ensure RevCam has permission to manage NetworkManager."
                ) from exc
            error_message = f"Unable to enable hotspot {cleaned_ssid}: {message}."
            self._record_log(
                "hotspot_enable_error",
                error_message,
                metadata=attempt_metadata,
            )
            raise WiFiError(f"Unable to enable hotspot: {message}") from exc
        self._hotspot_profile = status.profile or self._hotspot_profile
        self._update_mdns(status)
        effective_timeout = None
        if development_mode:
            base_timeout = (
                self._hotspot_rollback_timeout
                if rollback_timeout is None
                else max(0.0, rollback_timeout)
            )
            effective_timeout = base_timeout
        result_metadata = dict(attempt_metadata)
        if effective_timeout is not None:
            result_metadata["effective_rollback"] = effective_timeout
        monitor_required = False
        if status.hotspot_active:
            message = f"Hotspot {status.ssid or cleaned_ssid} enabled."
            self._record_log(
                "hotspot_enabled",
                message,
                status=status,
                metadata=result_metadata,
            )
            if not development_mode:
                return status
            timeout = effective_timeout
            if timeout is None or timeout <= 0:
                return status
            monitor_required = False
        else:
            detail = status.detail or "Hotspot activation pending."
            message = f"Hotspot {cleaned_ssid} pending: {detail}"
            self._record_log(
                "hotspot_status",
                message,
                status=status,
                metadata=result_metadata,
            )
            if not development_mode:
                return status
            timeout = (
                effective_timeout
                if effective_timeout is not None
                else self._hotspot_rollback_timeout
                if rollback_timeout is None
                else max(0.0, rollback_timeout)
            )
            monitor_required = True
        if timeout <= 0 or not monitor_required:
            return status
        self._record_log(
            "hotspot_monitor",
            f"Monitoring hotspot {cleaned_ssid} for up to {int(math.ceil(timeout))}s before rollback.",
            status=status,
            metadata=result_metadata,
        )
        deadline = time.monotonic() + timeout
        current = status
        while time.monotonic() < deadline:
            if current.hotspot_active:
                self._hotspot_profile = current.profile or self._hotspot_profile
                self._update_mdns(current)
                self._record_log(
                    "hotspot_confirmed",
                    f"Hotspot {cleaned_ssid} became active before rollback deadline.",
                    status=current,
                    metadata=result_metadata,
                )
                return current
            time.sleep(self._poll_interval)
            current = self._fetch_status()
        if previous_profile:
            restored = self._backend.activate_profile(previous_profile)
            previous_name = (
                previous_status.ssid if previous_status and previous_status.ssid else previous_profile
            )
            restored.detail = (
                f"Hotspot {cleaned_ssid} did not become active within {int(math.ceil(timeout))}s; "
                f"restored {previous_name}."
            )
            self._update_mdns(restored)
            if previous_status and previous_status.hotspot_active:
                self._hotspot_profile = previous_profile
            elif not restored.hotspot_active:
                self._hotspot_profile = None
            rollback_metadata = dict(result_metadata)
            rollback_metadata["restored_profile"] = previous_profile
            self._record_log(
                "hotspot_rollback",
                restored.detail,
                status=restored,
                metadata=rollback_metadata,
            )
            return restored
        current.detail = (
            f"Hotspot {cleaned_ssid} did not become active within {int(math.ceil(timeout))}s and "
            "no previous connection was available to restore."
        )
        self._update_mdns(current)
        self._record_log(
            "hotspot_timeout",
            current.detail,
            status=current,
            metadata=result_metadata,
        )
        return current

    def disable_hotspot(self) -> WiFiStatus:
        metadata = {
            "target": self._hotspot_profile,
        }
        self._record_log("hotspot_disable_attempt", "Disabling hotspot.", metadata=metadata)
        try:
            status = self._backend.stop_hotspot(self._hotspot_profile)
        except WiFiError as exc:
            message = str(exc).strip() or "Unable to disable hotspot"
            self._record_log(
                "hotspot_disable_error",
                f"Hotspot disable failed: {message}.",
                metadata=metadata,
            )
            raise
        self._update_mdns(status)
        if not status.hotspot_active:
            self._hotspot_profile = None
        detail = status.detail or "Hotspot disabled."
        self._record_log(
            "hotspot_disabled",
            detail,
            status=status,
            metadata=metadata,
        )
        return status

    def close(self) -> None:
        if self._mdns is not None:
            try:
                self._mdns.close()
            except Exception:  # pragma: no cover - best effort cleanup
                logging.getLogger(__name__).debug("Error shutting down mDNS advertiser", exc_info=True)
            finally:
                self._mdns = None

    # ------------------------------ helpers -----------------------------
    def _fetch_status(self) -> WiFiStatus:
        status = self._backend.get_status()
        self._update_mdns(status)
        return status

    def get_connection_log(self, limit: int | None = None) -> list[dict[str, object | None]]:
        """Return recent Wi-Fi connection and hotspot events."""

        with self._log_lock:
            entries = list(self._log)
        if limit is not None:
            try:
                limit_value = max(1, int(limit))
            except (TypeError, ValueError):
                limit_value = 1
            if len(entries) > limit_value:
                entries = entries[-limit_value:]
        return [entry.to_dict() for entry in reversed(entries)]

    def _record_log(
        self,
        event: str,
        message: str,
        *,
        status: WiFiStatus | None = None,
        metadata: dict[str, object | None] | None = None,
    ) -> None:
        """Store a troubleshooting entry and mirror it to the logger."""

        status_payload: dict[str, object | None] | None
        if isinstance(status, WiFiStatus):
            status_payload = status.to_dict()
        else:
            status_payload = None
        metadata_payload: dict[str, object | None] | None
        if metadata:
            metadata_payload = {
                key: value
                for key, value in metadata.items()
                if value is not None
            }
        else:
            metadata_payload = None
        entry = WiFiLogEntry(
            timestamp=time.time(),
            event=event,
            message=message,
            status=status_payload,
            metadata=metadata_payload,
        )
        with self._log_lock:
            self._log.append(entry)
        logger = logging.getLogger(__name__)
        if metadata_payload:
            logger.info("Wi-Fi event %s: %s | metadata=%s", event, message, metadata_payload)
        else:
            logger.info("Wi-Fi event %s: %s", event, message)

    def _update_mdns(self, status: WiFiStatus) -> None:
        if self._mdns is None:
            return
        try:
            self._mdns.advertise(status.ip_address)
        except Exception:  # pragma: no cover - best effort logging
            logging.getLogger(__name__).debug("mDNS advertise failed", exc_info=True)


__all__ = [
    "WiFiError",
    "WiFiNetwork",
    "WiFiStatus",
    "WiFiBackend",
    "NMCLIBackend",
    "WiFiManager",
    "WiFiLogEntry",
]
