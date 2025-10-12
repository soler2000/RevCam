"""Wi-Fi management utilities with safe fallbacks for RevCam."""

from __future__ import annotations

import logging
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, Sequence

from .mdns import MDNSAdvertiser
from .system_log import SystemLog, SystemLogEntry


DEFAULT_HOTSPOT_PASSWORD = "Reversing123"


class WiFiCredentialStore:
    """Persist Wi-Fi credentials to disk for reuse across restarts."""

    def __init__(self, path: Path | str = Path("data/wifi_credentials.json")) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._hotspot_password: str | None = None
        self._network_passwords: dict[str, str] = {}
        self._ensure_parent()
        self._load()

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).warning("Unable to load Wi-Fi credentials: %s", exc)
            return
        hotspot_password = payload.get("hotspot_password")
        if isinstance(hotspot_password, str) and hotspot_password:
            self._hotspot_password = hotspot_password
        else:
            self._hotspot_password = None
        networks = payload.get("networks")
        if isinstance(networks, dict):
            cleaned: dict[str, str] = {}
            for raw_ssid, raw_password in networks.items():
                if not isinstance(raw_ssid, str):
                    continue
                if not isinstance(raw_password, str) or not raw_password:
                    continue
                ssid = raw_ssid.strip()
                if not ssid:
                    continue
                cleaned[ssid] = raw_password
            self._network_passwords = cleaned
        else:
            self._network_passwords = {}

    def _save_locked(self) -> None:
        payload = {
            "hotspot_password": self._hotspot_password,
            "networks": dict(self._network_passwords),
        }
        try:
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logging.getLogger(__name__).warning("Unable to persist Wi-Fi credentials: %s", exc)

    def get_hotspot_password(self) -> str | None:
        with self._lock:
            return self._hotspot_password

    def set_hotspot_password(self, password: str | None) -> None:
        cleaned = password.strip() if isinstance(password, str) and password.strip() else None
        with self._lock:
            self._hotspot_password = cleaned
            self._save_locked()

    def get_network_password(self, ssid: str) -> str | None:
        if not isinstance(ssid, str):
            return None
        identifier = ssid.strip()
        if not identifier:
            return None
        with self._lock:
            return self._network_passwords.get(identifier)

    def list_networks(self) -> list[str]:
        """Return the SSIDs with stored credentials."""

        with self._lock:
            return sorted(self._network_passwords)

    def set_network_password(self, ssid: str, password: str | None) -> None:
        if not isinstance(ssid, str) or not ssid.strip():
            raise ValueError("SSID must be a non-empty string")
        identifier = ssid.strip()
        cleaned_password = password.strip() if isinstance(password, str) and password.strip() else None
        with self._lock:
            if cleaned_password is None:
                self._network_passwords.pop(identifier, None)
            else:
                self._network_passwords[identifier] = cleaned_password
            self._save_locked()

    def forget_network(self, ssid: str) -> None:
        if not isinstance(ssid, str) or not ssid.strip():
            return
        identifier = ssid.strip()
        with self._lock:
            if identifier in self._network_passwords:
                self._network_passwords.pop(identifier, None)
                self._save_locked()


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
    stored_credentials: bool = False

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
            "stored_credentials": self.stored_credentials,
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
    hotspot_password: str | None = None

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
            "hotspot_password": self.hotspot_password,
        }


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

    def disconnect(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def forget(self, profile_or_ssid: str) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def hotspot_diagnostics(self) -> dict[str, object | None] | None:  # pragma: no cover - optional hook
        """Return backend-specific hotspot troubleshooting details when available."""

        return None


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
        password_provided = bool(password)
        attempt_history: list[dict[str, object | None]] | None = None
        try:
            attempt_history = self._prepare_hotspot_profile(
                interface,
                connection_name,
                ssid,
                password if password_provided else None,
            )
            try:
                self._run(["nmcli", "connection", "down", connection_name])
            except WiFiError:
                # Connection might not be active yet; ignore and proceed.
                pass
            self._run(["nmcli", "connection", "up", connection_name])
        except WiFiError as exc:
            if (
                attempt_history
                and not hasattr(exc, "details")
                and isinstance(attempt_history, list)
            ):
                setattr(
                    exc,
                    "details",
                    {
                        "connection": connection_name,
                        "attempts": attempt_history,
                    },
                )
            descriptor = "secure" if password_provided else "open"
            wrapped = WiFiError(f"Unable to configure {descriptor} hotspot: {exc}")
            if hasattr(exc, "details"):
                setattr(wrapped, "details", getattr(exc, "details"))
            raise wrapped from exc
        status = self.get_status()
        status.profile = connection_name
        status.mode = "access-point"
        status.hotspot_active = True
        self._last_hotspot_profile = connection_name
        return status

    def _clear_security_property(
        self,
        connection_name: str,
        property_name: str,
        *,
        fallback_value: str | None = "",
    ) -> tuple[bool, list[dict[str, object | None]]]:
        """Remove a security property for the given connection.

        Returns a tuple describing whether the property could be cleared and a
        list of step dictionaries suitable for troubleshooting metadata.
        """

        logger = logging.getLogger(__name__)
        steps: list[dict[str, object | None]] = []
        removal_success = False
        try:
            self._run(
                [
                    "nmcli",
                    "connection",
                    "modify",
                    connection_name,
                    f"-{property_name}",
                ]
            )
        except WiFiError as exc:
            message = str(exc).strip() or None
            if self._is_missing_security_setting_error(message):
                steps.append(
                    {
                        "property": property_name,
                        "action": "remove",
                        "result": "missing",
                        "error": message,
                    }
                )
                logger.debug(
                    "Hotspot security property %s already absent on %s: %s",
                    property_name,
                    connection_name,
                    message,
                )
                removal_success = True
            else:
                steps.append(
                    {
                        "property": property_name,
                        "action": "remove",
                        "result": "error",
                        "error": message,
                    }
                )
                logger.debug(
                    "Unable to remove hotspot security property %s from %s: %s",
                    property_name,
                    connection_name,
                    message,
                )
        else:
            steps.append(
                {
                    "property": property_name,
                    "action": "remove",
                    "result": "success",
                }
            )
            removal_success = True
        if fallback_value is None:
            return removal_success, steps
        try:
            self._run(
                [
                    "nmcli",
                    "connection",
                    "modify",
                    connection_name,
                    property_name,
                    fallback_value,
                ]
            )
        except WiFiError as exc:
            message = str(exc).strip() or None
            steps.append(
                {
                    "property": property_name,
                    "action": "set",
                    "result": "error",
                    "value": fallback_value,
                    "error": message,
                }
            )
            logger.debug(
                "Unable to set hotspot security fallback %s=%s on %s: %s",
                property_name,
                fallback_value,
                connection_name,
                message,
            )
            return False, steps
        steps.append(
            {
                "property": property_name,
                "action": "set",
                "result": "success",
                "value": fallback_value,
            }
        )
        return True, steps

    @staticmethod
    def _is_missing_security_property_error(message: str | None) -> bool:
        if not message:
            return False
        normalized = message.strip().lower()
        if (
            "value for '-802-11-wireless-security" in normalized
            and "missing" in normalized
        ):
            return True
        return any(
            token in normalized
            for token in (
                "no such property",
                "property not found",
                "unknown property",
                "property doesn't exist",
                "property does not exist",
                "property is not found",
            )
        )

    @staticmethod
    def _is_missing_security_setting_error(message: str | None) -> bool:
        if not message:
            return False
        normalized = message.strip().lower()
        if NMCLIBackend._is_missing_security_property_error(message):
            return True
        if "invalid <setting>." in normalized and "802-11-wireless-security" in normalized:
            return True
        return False

    def _fetch_connection_security_fields(
        self, connection_name: str
    ) -> tuple[dict[str, str] | None, str | None]:
        """Return raw security fields for a connection or an error message."""

        try:
            output = self._run(
                [
                    "nmcli",
                    "--show-secrets",
                    "yes",
                    "-g",
                    ",".join(
                        [
                            "802-11-wireless-security.key-mgmt",
                            "802-11-wireless-security.psk",
                            "802-11-wireless-security.wep-key0",
                            "802-11-wireless-security.wep-key1",
                            "802-11-wireless-security.wep-key2",
                            "802-11-wireless-security.wep-key3",
                            "802-11-wireless-security.wep-key-flags",
                        ]
                    ),
                    "connection",
                    "show",
                    connection_name,
                ]
            )
        except WiFiError as exc:
            return None, str(exc).strip() or str(exc)
        fields = output.splitlines()
        values: dict[str, str] = {}
        keys = [
            "802-11-wireless-security.key-mgmt",
            "802-11-wireless-security.psk",
            "802-11-wireless-security.wep-key0",
            "802-11-wireless-security.wep-key1",
            "802-11-wireless-security.wep-key2",
            "802-11-wireless-security.wep-key3",
            "802-11-wireless-security.wep-key-flags",
        ]
        for key, value in zip(keys, fields):
            values[key] = value.strip()
        return values, None

    @staticmethod
    def _security_values_require_secret(values: dict[str, str]) -> bool:
        key_mgmt = values.get("802-11-wireless-security.key-mgmt", "")
        if key_mgmt and key_mgmt != "none":
            return True
        for secret_key in (
            "802-11-wireless-security.psk",
            "802-11-wireless-security.wep-key0",
            "802-11-wireless-security.wep-key1",
            "802-11-wireless-security.wep-key2",
            "802-11-wireless-security.wep-key3",
        ):
            if values.get(secret_key):
                return True
        flags_value = values.get("802-11-wireless-security.wep-key-flags", "")
        if flags_value:
            normalized = flags_value.strip().split()[0]
            if normalized:
                if normalized.lower().startswith("0x"):
                    try:
                        if int(normalized, 16) != 0:
                            return True
                    except ValueError:
                        return True
                else:
                    try:
                        if int(normalized, 10) != 0:
                            return True
                    except ValueError:
                        if normalized not in {"0", ""}:
                            return True
        return False

    def _connection_security_requires_secret(self, connection_name: str) -> bool:
        """Return True when the connection still expects a Wi-Fi secret."""

        values, error = self._fetch_connection_security_fields(connection_name)
        if not values or error:
            return False
        return self._security_values_require_secret(values)

    def _reset_hotspot_security(
        self, connection_name: str
    ) -> tuple[bool, list[dict[str, object | None]]]:
        """Ensure the hotspot connection no longer expects a secret."""

        logger = logging.getLogger(__name__)
        attempts: list[dict[str, object | None]] = []
        removed, removal_step = self._remove_security_setting(connection_name)
        removal_with_context = dict(removal_step)
        removal_with_context.setdefault("connection", connection_name)
        attempts.append(removal_with_context)
        cleared_any = False
        for property_name, fallback in (
            ("802-11-wireless-security.auth-alg", "open"),
            ("802-11-wireless-security.psk", ""),
            ("802-11-wireless-security.psk-flags", "0"),
            ("802-11-wireless-security.wep-key0", ""),
            ("802-11-wireless-security.wep-key1", ""),
            ("802-11-wireless-security.wep-key2", ""),
            ("802-11-wireless-security.wep-key3", ""),
            ("802-11-wireless-security.wep-key-flags", "0"),
            ("802-11-wireless-security.wep-key-type", None),
            ("802-11-wireless-security.wep-tx-keyidx", "0"),
        ):
            cleared, steps = self._clear_security_property(
                connection_name, property_name, fallback_value=fallback
            )
            for step in steps:
                step_with_context = dict(step)
                step_with_context.setdefault("connection", connection_name)
                attempts.append(step_with_context)
            if cleared:
                cleared_any = True
        success = removed or cleared_any
        try:
            self._run(
                [
                    "nmcli",
                    "connection",
                    "modify",
                    connection_name,
                    "802-11-wireless-security.key-mgmt",
                    "none",
                ]
            )
        except WiFiError as exc:
            message = str(exc).strip() or None
            attempt: dict[str, object | None] = {
                "connection": connection_name,
                "property": "802-11-wireless-security.key-mgmt",
                "action": "set",
                "value": "none",
                "result": "error",
                "error": message,
            }
            if self._is_missing_connection_error(message):
                attempt["hint"] = "missing_connection"
            attempts.append(attempt)
            logger.debug(
                "Unable to update hotspot key management to none: %s", message
            )
        else:
            attempts.append(
                {
                    "connection": connection_name,
                    "property": "802-11-wireless-security.key-mgmt",
                    "action": "set",
                    "value": "none",
                    "result": "success",
                }
            )
            success = True
        values, error = self._fetch_connection_security_fields(connection_name)
        if error:
            attempts.append(
                {
                    "connection": connection_name,
                    "action": "inspect",
                    "result": "error",
                    "error": error,
                }
            )
            return success, attempts
        requires_secret = self._security_values_require_secret(values)
        inspect_step: dict[str, object | None] = {
            "connection": connection_name,
            "action": "inspect",
            "values": values,
            "requires_secret": requires_secret,
        }
        if requires_secret:
            inspect_step["result"] = "error"
            inspect_step["error"] = "hotspot security still expects secrets"
            logger.debug(
                "Hotspot connection %s still references secrets after reset",
                connection_name,
            )
            attempts.append(inspect_step)
            return False, attempts
        inspect_step["result"] = "success"
        attempts.append(inspect_step)
        return True, attempts

    @staticmethod
    def _is_missing_connection_error(message: str | None) -> bool:
        if not message:
            return False
        normalized = message.strip().lower()
        for token in (
            "unknown connection",
            "no such connection",
            "not find connection",
            "cannot find connection",
            "does not exist",
            "not exist",
        ):
            if token in normalized:
                return True
        return False

    def _remove_security_setting(
        self, connection_name: str
    ) -> tuple[bool, dict[str, object | None]]:
        """Remove the entire security setting from the connection."""

        logger = logging.getLogger(__name__)
        try:
            self._run(
                [
                    "nmcli",
                    "connection",
                    "modify",
                    connection_name,
                    "-802-11-wireless-security",
                ]
            )
        except WiFiError as exc:
            message = str(exc).strip() or None
            if self._is_missing_security_property_error(message):
                logger.debug(
                    "Hotspot security setting already absent from %s: %s",
                    connection_name,
                    message,
                )
                return True, {
                    "action": "remove_setting",
                    "result": "missing",
                    "error": message,
                }
            logger.debug(
                "Unable to remove hotspot security setting from %s: %s",
                connection_name,
                message,
            )
            return False, {"action": "remove_setting", "result": "error", "error": message}
        return True, {"action": "remove_setting", "result": "success"}

    def hotspot_diagnostics(self) -> dict[str, object | None] | None:
        """Return current hotspot security fields for troubleshooting."""

        connection_name = self._last_hotspot_profile or "RevCam Hotspot"
        values, error = self._fetch_connection_security_fields(connection_name)
        diagnostics: dict[str, object | None] = {"connection": connection_name}
        if error:
            diagnostics["error"] = error
            return diagnostics
        if values is None:
            diagnostics["values"] = None
            return diagnostics
        diagnostics["values"] = values
        diagnostics["requires_secret"] = self._security_values_require_secret(values)
        return diagnostics

    def _prepare_hotspot_profile(
        self,
        interface: str,
        connection_name: str,
        ssid: str,
        password: str | None,
    ) -> list[dict[str, object | None]]:
        """Rebuild the hotspot profile with shared networking."""

        profile_steps: list[dict[str, object | None]] = []
        configure_steps: list[dict[str, object | None]] = []
        security_steps: list[dict[str, object | None]] = []

        def _clone(steps: list[dict[str, object | None]]) -> list[dict[str, object | None]]:
            return [dict(step) for step in steps]

        def _build_attempts() -> list[dict[str, object | None]]:
            attempts: list[dict[str, object | None]] = []
            if profile_steps:
                attempts.append({"stage": "profile", "steps": _clone(profile_steps)})
            if configure_steps:
                attempts.append(
                    {"stage": "configure", "steps": _clone(configure_steps)}
                )
            if security_steps:
                attempts.append(
                    {"stage": "initial", "steps": _clone(security_steps)}
                )
            return attempts

        delete_command = ["nmcli", "connection", "delete", connection_name]
        try:
            self._run(delete_command)
        except WiFiError as exc:
            message = str(exc).strip() or None
            step: dict[str, object | None] = {
                "action": "delete_connection",
                "command": delete_command,
                "result": "error",
                "error": message,
            }
            if self._is_missing_connection_error(message):
                step["result"] = "missing"
            profile_steps.append(step)
            if step["result"] == "error":
                setattr(
                    exc,
                    "details",
                    {"connection": connection_name, "attempts": _build_attempts()},
                )
                raise
        else:
            profile_steps.append(
                {
                    "action": "delete_connection",
                    "command": delete_command,
                    "result": "success",
                }
            )

        add_command = [
            "nmcli",
            "connection",
            "add",
            "type",
            "wifi",
            "ifname",
            interface,
            "con-name",
            connection_name,
            "autoconnect",
            "yes",
            "ssid",
            ssid,
        ]
        if password:
            add_command.extend(["wifi-sec.key-mgmt", "wpa-psk"])
        else:
            add_command.extend(["wifi-sec.key-mgmt", "none"])
        try:
            self._run(add_command)
        except WiFiError as exc:
            message = str(exc).strip() or None
            profile_steps.append(
                {
                    "action": "add_connection",
                    "command": add_command,
                    "result": "error",
                    "error": message,
                }
            )
            setattr(
                exc,
                "details",
                {"connection": connection_name, "attempts": _build_attempts()},
            )
            raise
        else:
            profile_steps.append(
                {
                    "action": "add_connection",
                    "command": add_command,
                    "result": "success",
                }
            )

        base_settings = [
            ("connection.interface-name", interface),
            ("wifi.ssid", ssid),
            ("802-11-wireless.mode", "ap"),
            ("ipv4.method", "shared"),
            ("ipv6.method", "ignore"),
            ("connection.autoconnect", "yes"),
        ]
        for property_name, value in base_settings:
            command = [
                "nmcli",
                "connection",
                "modify",
                connection_name,
                property_name,
                value,
            ]
            try:
                self._run(command)
            except WiFiError as exc:
                message = str(exc).strip() or None
                configure_steps.append(
                    {
                        "action": "set",
                        "property": property_name,
                        "value": value,
                        "result": "error",
                        "error": message,
                    }
                )
                attempts = _build_attempts()
                setattr(
                    exc,
                    "details",
                    {"connection": connection_name, "attempts": attempts},
                )
                raise
            else:
                configure_steps.append(
                    {
                        "action": "set",
                        "property": property_name,
                        "value": value,
                        "result": "success",
                    }
                )

        if password:
            security_settings = [
                ("802-11-wireless-security.key-mgmt", "wpa-psk"),
                ("802-11-wireless-security.auth-alg", "open"),
                ("802-11-wireless-security.proto", "rsn"),
                ("802-11-wireless-security.group", "ccmp"),
                ("802-11-wireless-security.pairwise", "ccmp"),
                ("802-11-wireless-security.psk", password),
                ("802-11-wireless-security.psk-flags", "0"),
            ]
            for property_name, value in security_settings:
                command = [
                    "nmcli",
                    "connection",
                    "modify",
                    connection_name,
                    property_name,
                    value,
                ]
                try:
                    self._run(command)
                except WiFiError as exc:
                    message = str(exc).strip() or None
                    security_steps.append(
                        {
                            "action": "set",
                            "property": property_name,
                            "value": "<hidden>"
                            if property_name.endswith(".psk")
                            else value,
                            "result": "error",
                            "error": message,
                        }
                    )
                    attempts = _build_attempts()
                    setattr(
                        exc,
                        "details",
                        {"connection": connection_name, "attempts": attempts},
                    )
                    raise
                else:
                    security_steps.append(
                        {
                            "action": "set",
                            "property": property_name,
                            "value": "<hidden>"
                            if property_name.endswith(".psk")
                            else value,
                            "result": "success",
                        }
                    )
        else:
            security_cleared, security_attempts = self._reset_hotspot_security(
                connection_name
            )
            security_steps.extend(security_attempts)
            if not security_cleared:
                attempts = _build_attempts()
                failure = WiFiError("Unable to clear hotspot security configuration")
                setattr(
                    failure,
                    "details",
                    {"connection": connection_name, "attempts": attempts},
                )
                raise failure

        return _build_attempts()

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

    def disconnect(self) -> None:
        interface = self._get_interface()
        try:
            self._run(["nmcli", "device", "disconnect", interface])
        except WiFiError as exc:
            message = str(exc).strip()
            lowered = message.lower()
            if "not authorized" in lowered or "not authorised" in lowered:
                raise WiFiError(
                    "Unable to disconnect from Wi-Fi: not authorized to control networking. "
                    "Ensure RevCam has permission to manage NetworkManager."
                ) from exc
            if "is not active" in lowered or "already disconnected" in lowered:
                return
            raise

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
            if any(
                phrase in lowered
                for phrase in (
                    "unknown connection",
                    "cannot delete unknown connection",
                    "no such connection",
                )
            ):
                logging.getLogger(__name__).info(
                    "Requested removal of unknown Wi-Fi connection \"%s\": %s",
                    profile_or_ssid,
                    message,
                )
                return
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
        default_hotspot_ssid: str = "RevCam",
        default_hotspot_password: str = DEFAULT_HOTSPOT_PASSWORD,
        credential_store: WiFiCredentialStore | None = None,
        watchdog_boot_delay: float = 30.0,
        watchdog_interval: float = 30.0,
        watchdog_retry_delay: float = 5.0,
        log_path: Path | str | None = Path("data/system_log.jsonl"),
        system_log: SystemLog | None = None,
    ) -> None:
        self._backend = backend or NMCLIBackend()
        self._rollback_timeout = rollback_timeout
        self._poll_interval = max(0.1, poll_interval)
        self._hotspot_profile: str | None = None
        self._credentials = credential_store or WiFiCredentialStore()
        cleaned_default_password = (
            default_hotspot_password.strip()
            if isinstance(default_hotspot_password, str)
            else ""
        )
        self._default_hotspot_password = (
            cleaned_default_password or DEFAULT_HOTSPOT_PASSWORD
        )
        stored_password = self._credentials.get_hotspot_password()
        if isinstance(stored_password, str) and stored_password.strip():
            self._hotspot_password = stored_password.strip()
        else:
            self._hotspot_password = self._default_hotspot_password
            if stored_password != self._hotspot_password:
                self._credentials.set_hotspot_password(self._hotspot_password)
        self._hotspot_rollback_timeout = (
            self._rollback_timeout
            if hotspot_rollback_timeout is None
            else max(0.0, hotspot_rollback_timeout)
        )
        self._mdns = mdns_advertiser
        cleaned_default = default_hotspot_ssid.strip() if isinstance(default_hotspot_ssid, str) else ""
        self._default_hotspot_ssid = cleaned_default or "RevCam"
        if self._mdns is None:
            try:
                self._mdns = MDNSAdvertiser()
            except Exception as exc:  # pragma: no cover - environment specific
                logging.getLogger(__name__).warning(
                    "mDNS advertising disabled: %s", exc
                )
                self._mdns = None
        self._log: Deque[SystemLogEntry] = deque(maxlen=200)
        self._log_lock = threading.Lock()
        if isinstance(system_log, SystemLog):
            self._system_log = system_log
        else:
            self._system_log = SystemLog(path=log_path, max_entries=1000)
        self._restore_network_log()
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop_event: threading.Event | None = None
        self._watchdog_boot_delay = max(0.0, watchdog_boot_delay)
        self._watchdog_interval = max(0.0, watchdog_interval)
        self._watchdog_retry_delay = max(0.0, watchdog_retry_delay)
        self._status_listener: Callable[[WiFiStatus], None] | None = None
        try:
            initial_status = self._backend.get_status()
        except WiFiError:
            initial_status = None
        if initial_status:
            self._finalise_status(initial_status)

    @property
    def system_log(self) -> SystemLog:
        """Expose the shared system log instance."""

        return self._system_log

    # ------------------------------ operations -----------------------------
    def get_status(self) -> WiFiStatus:
        return self._fetch_status()

    def scan_networks(self) -> Sequence[WiFiNetwork]:
        networks = list(self._backend.scan())
        stored = {
            ssid.strip()
            for ssid in self._credentials.list_networks()
            if isinstance(ssid, str) and ssid.strip()
        }
        if stored:
            for network in networks:
                if network.hidden:
                    continue
                identifier = network.ssid.strip()
                if not identifier:
                    continue
                if identifier in stored:
                    network.stored_credentials = True
                    network.known = True
        return networks

    def auto_connect_known_networks(
        self,
        *,
        start_hotspot: bool = True,
        update_mdns: bool = True,
    ) -> WiFiStatus:
        """Join the strongest known network or fall back to hotspot mode."""

        def _signal_value(network: WiFiNetwork) -> float:
            if network.signal is None:
                return float("-inf")
            return float(network.signal)

        known_ssids = {
            ssid.strip()
            for ssid in self._credentials.list_networks()
            if isinstance(ssid, str) and ssid.strip()
        }

        try:
            initial_status = self._fetch_status(update_mdns=update_mdns)
        except WiFiError as exc:
            initial_status = None
            self._record_log(
                "auto_connect_status_error",
                f"Unable to refresh Wi-Fi status before auto-connect: {exc}.",
            )
        else:
            if initial_status.connected and not initial_status.hotspot_active:
                identifiers = {
                    value.strip()
                    for value in (initial_status.profile, initial_status.ssid)
                    if isinstance(value, str) and value.strip()
                }
                if identifiers & known_ssids:
                    self._record_log(
                        "auto_connect_skip",
                        "Wi-Fi already connected; skipping automatic network selection.",
                        status=initial_status,
                    )
                    return initial_status

        self._record_log(
            "auto_connect_start",
            "Scanning for known Wi-Fi networks.",
        )
        try:
            scan_results = list(self.scan_networks())
        except WiFiError as exc:
            scan_results = []
            self._record_log(
                "auto_connect_scan_error",
                f"Unable to scan Wi-Fi networks: {exc}.",
            )
        candidates: dict[str, WiFiNetwork] = {}
        for network in scan_results:
            ssid = network.ssid.strip()
            if not ssid or network.hidden:
                continue
            if not (network.known or ssid in known_ssids):
                continue
            existing = candidates.get(ssid)
            if existing is None or _signal_value(network) > _signal_value(existing):
                candidates[ssid] = network

        if not candidates:
            self._record_log(
                "auto_connect_no_candidates",
                "No known Wi-Fi networks detected during automatic selection.",
            )
        last_status = initial_status
        for ssid, network in sorted(
            candidates.items(),
            key=lambda item: _signal_value(item[1]),
            reverse=True,
        ):
            metadata = {
                "signal": network.signal,
                "known": network.known,
                "security": network.security,
            }
            self._record_log(
                "auto_connect_candidate",
                f"Attempting automatic connection to {ssid}.",
                metadata=metadata,
            )
            try:
                status = self.connect(ssid)
            except WiFiError as exc:
                self._record_log(
                    "auto_connect_error",
                    f"Automatic connection to {ssid} failed: {exc}.",
                    metadata=metadata,
                )
                continue
            last_status = status
            if status.connected and not status.hotspot_active and (status.ssid == ssid or status.profile == ssid):
                self._record_log(
                    "auto_connect_success",
                    f"Automatically connected to {status.ssid or ssid}.",
                    status=status,
                    metadata=metadata,
                )
                return status
            self._record_log(
                "auto_connect_incomplete",
                f"{ssid} did not report an active connection; continuing to next candidate.",
                status=status,
                metadata=metadata,
            )

        if not start_hotspot:
            return last_status or self._fetch_status(update_mdns=update_mdns)

        self._record_log(
            "auto_connect_hotspot_fallback",
            "No known networks reachable; enabling hotspot mode.",
        )
        try:
            status = self.enable_hotspot()
        except WiFiError as exc:
            self._record_log(
                "auto_connect_hotspot_error",
                f"Unable to enable hotspot as a fallback: {exc}.",
            )
            raise
        return status

    def forget_network(self, profile_or_ssid: str) -> WiFiStatus:
        if not isinstance(profile_or_ssid, str) or not profile_or_ssid.strip():
            raise WiFiError("Network identifier must be a non-empty string")
        identifier = profile_or_ssid.strip()
        try:
            current_status = self._backend.get_status()
        except WiFiError:
            current_status = None
        self._backend.forget(identifier)
        if identifier == self._hotspot_profile:
            self._hotspot_profile = None
        self._credentials.forget_network(identifier)
        should_disconnect = False
        if isinstance(current_status, WiFiStatus):
            if current_status.connected and not current_status.hotspot_active:
                active_identifiers = {
                    value.strip()
                    for value in (current_status.ssid, current_status.profile)
                    if isinstance(value, str) and value.strip()
                }
                if identifier in active_identifiers:
                    should_disconnect = True
        if should_disconnect:
            self._backend.disconnect()
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
        supplied_password = cleaned_password if password is not None else None
        stored_credentials_used = False
        if cleaned_password is None:
            stored_password = self._credentials.get_network_password(cleaned_ssid)
            if stored_password:
                cleaned_password = stored_password
                stored_credentials_used = True
        attempt_metadata: dict[str, object | None] = {
            "target": cleaned_ssid,
            "development_mode": development_mode,
        }
        if rollback_timeout is not None:
            attempt_metadata["requested_rollback"] = max(0.0, rollback_timeout)
        attempt_metadata["password_provided"] = cleaned_password is not None
        attempt_metadata["used_stored_credentials"] = stored_credentials_used
        self._record_log(
            "connect_attempt",
            f"Attempting to connect to {cleaned_ssid}.",
            metadata=attempt_metadata,
        )
        previous_status = self._fetch_status(update_mdns=False)
        previous_profile = previous_status.profile if previous_status.connected else None
        previous_identifier: str | None = None
        if previous_status.connected:
            prior_value = previous_status.ssid or previous_status.profile
            if isinstance(prior_value, str) and prior_value.strip():
                previous_identifier = prior_value.strip()
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
        if supplied_password is not None:
            try:
                self._credentials.set_network_password(cleaned_ssid, supplied_password)
            except ValueError:
                pass
        if status.connected and not status.hotspot_active and not status.ip_address:
            sync_timeout = min(0.5, max(self._poll_interval * 2, 0.1))
            awaited = self._await_station_ip(
                initial_status=status,
                timeout=sync_timeout,
                sleep_interval=self._poll_interval,
            )
            if isinstance(awaited, WiFiStatus):
                status = awaited
            if status.connected and not status.hotspot_active and not status.ip_address:
                self._schedule_ip_refresh(initial_status=status)
        rollback_requested = development_mode or (rollback_timeout is not None)
        effective_timeout = None
        if development_mode:
            base_timeout = (
                self._rollback_timeout if rollback_timeout is None else max(0.0, rollback_timeout)
            )
            effective_timeout = base_timeout
        elif rollback_timeout is not None:
            effective_timeout = max(0.0, rollback_timeout)
        result_metadata = dict(attempt_metadata)
        if effective_timeout is not None:
            result_metadata["effective_rollback"] = effective_timeout
        monitor_required = False
        if status.connected and (status.ssid == cleaned_ssid or status.profile == cleaned_ssid):
            reminder_text: str | None = None
            current_identifier = status.ssid or status.profile or cleaned_ssid
            if isinstance(current_identifier, str):
                current_identifier = current_identifier.strip() or cleaned_ssid
            if (
                isinstance(current_identifier, str)
                and current_identifier
                and previous_identifier
                and current_identifier != previous_identifier
            ):
                reminder_text = (
                    "Remember to reconnect your controlling device to "
                    f'"{current_identifier}" to continue.'
                )
            if reminder_text:
                existing_detail = status.detail.strip() if isinstance(status.detail, str) else ""
                if existing_detail and not existing_detail.endswith("."):
                    existing_detail = f"{existing_detail}."
                status.detail = f"{existing_detail} {reminder_text}".strip()
                result_metadata["previous_connection"] = previous_identifier
            message = f"Connected to {status.ssid or cleaned_ssid}."
            self._record_log("connect_success", message, status=status, metadata=result_metadata)
            status = self._finalise_status(status)
            if not rollback_requested:
                return status
            timeout = effective_timeout
            if timeout is None or timeout <= 0:
                return status
            monitor_required = False
        else:
            detail = status.detail or "Connection did not report as active."
            message = f"Connection to {cleaned_ssid} pending: {detail}"
            self._record_log("connect_status", message, status=status, metadata=result_metadata)
            status = self._finalise_status(status)
            if not rollback_requested:
                return status
            timeout = effective_timeout
            if timeout is None and development_mode:
                timeout = self._rollback_timeout
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
            return self._finalise_status(status, update_mdns=False)
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
            current = self._fetch_status(update_mdns=False)
        restored = self._backend.activate_profile(previous_profile)
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
        return self._finalise_status(restored)

    def enable_hotspot(
        self,
        ssid: str | None = None,
        password: str | None = None,
        *,
        development_mode: bool = False,
        rollback_timeout: float | None = None,
    ) -> WiFiStatus:
        cleaned_ssid = ""
        if isinstance(ssid, str):
            cleaned_ssid = ssid.strip()
        cleaned_ssid = cleaned_ssid or self._default_hotspot_ssid
        provided_password = (
            password.strip() if isinstance(password, str) and password.strip() else None
        )
        if provided_password is not None and len(provided_password) < 8:
            raise WiFiError("Hotspot password must be at least 8 characters")
        if provided_password is None:
            stored_password = (
                self._hotspot_password.strip()
                if isinstance(self._hotspot_password, str)
                and self._hotspot_password.strip()
                else None
            )
            cleaned_password = stored_password or self._default_hotspot_password
        else:
            cleaned_password = provided_password
        if len(cleaned_password) < 8:
            raise WiFiError("Hotspot password must be at least 8 characters")
        attempt_metadata: dict[str, object | None] = {
            "target": cleaned_ssid,
            "development_mode": development_mode,
            "password_provided": provided_password is not None,
            "using_default_password": cleaned_password == self._default_hotspot_password,
            "default_ssid": cleaned_ssid == self._default_hotspot_ssid,
        }
        if rollback_timeout is not None:
            attempt_metadata["requested_rollback"] = max(0.0, rollback_timeout)
        self._record_log(
            "hotspot_enable_attempt",
            f"Enabling hotspot {cleaned_ssid}.",
            metadata=attempt_metadata,
        )
        try:
            previous_status = self._fetch_status(update_mdns=False)
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
            error_metadata = self._collect_hotspot_error_context(
                attempt_metadata, exc
            )
            self._record_log(
                "hotspot_enable_error",
                error_message,
                metadata=error_metadata if error_metadata else None,
            )
            failure = WiFiError(f"Unable to enable hotspot: {message}")
            details = getattr(exc, "details", None)
            if details is not None:
                setattr(failure, "details", details)
            raise failure from exc
        self._hotspot_password = cleaned_password
        self._credentials.set_hotspot_password(cleaned_password)
        self._hotspot_profile = status.profile or self._hotspot_profile
        rollback_requested = development_mode or (rollback_timeout is not None)
        effective_timeout = None
        if development_mode:
            base_timeout = (
                self._hotspot_rollback_timeout
                if rollback_timeout is None
                else max(0.0, rollback_timeout)
            )
            effective_timeout = base_timeout
        elif rollback_timeout is not None:
            effective_timeout = max(0.0, rollback_timeout)
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
            status = self._finalise_status(status)
            if not rollback_requested:
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
            status = self._finalise_status(status)
            if not rollback_requested:
                return status
            timeout = effective_timeout
            if timeout is None and development_mode:
                timeout = self._hotspot_rollback_timeout
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
            current = self._fetch_status(update_mdns=False)
        if previous_profile:
            restored = self._backend.activate_profile(previous_profile)
            previous_name = (
                previous_status.ssid if previous_status and previous_status.ssid else previous_profile
            )
            restored.detail = (
                f"Hotspot {cleaned_ssid} did not become active within {int(math.ceil(timeout))}s; "
                f"restored {previous_name}."
            )
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
            return self._finalise_status(restored)
        current.detail = (
            f"Hotspot {cleaned_ssid} did not become active within {int(math.ceil(timeout))}s and "
            "no previous connection was available to restore."
        )
        self._record_log(
            "hotspot_timeout",
            current.detail,
            status=current,
            metadata=result_metadata,
        )
        return self._finalise_status(current)

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
        if not status.hotspot_active:
            self._hotspot_profile = None
        detail = status.detail or "Hotspot disabled."
        self._record_log(
            "hotspot_disabled",
            detail,
            status=status,
            metadata=metadata,
        )
        return self._finalise_status(status)

    def close(self) -> None:
        self.stop_hotspot_watchdog()
        if self._mdns is not None:
            try:
                self._mdns.close()
            except Exception:  # pragma: no cover - best effort cleanup
                logging.getLogger(__name__).debug("Error shutting down mDNS advertiser", exc_info=True)
            finally:
                self._mdns = None

    # ------------------------------ helpers -----------------------------
    def start_hotspot_watchdog(self) -> None:
        """Begin monitoring connectivity for hotspot fallback."""

        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        stop_event = threading.Event()
        self._watchdog_stop_event = stop_event
        started_event = threading.Event()

        def _run() -> None:
            deadline = time.monotonic() + self._watchdog_boot_delay
            first_cycle = True
            while not stop_event.is_set():
                if first_cycle:
                    wait_time = max(0.0, deadline - time.monotonic())
                else:
                    wait_time = self._watchdog_interval
                if wait_time and stop_event.wait(wait_time):
                    break
                try:
                    self._watchdog_cycle(initial_boot_check=first_cycle)
                except Exception:  # pragma: no cover - defensive logging
                    logging.getLogger(__name__).exception("Wi-Fi hotspot watchdog error")
                finally:
                    if first_cycle:
                        started_event.set()
                first_cycle = False

        thread = threading.Thread(target=_run, name="wifi-hotspot-watchdog", daemon=True)
        self._watchdog_thread = thread
        thread.start()
        started_event.wait(timeout=min(self._watchdog_boot_delay + self._watchdog_interval, 0.5))

    def stop_hotspot_watchdog(self) -> None:
        """Stop the hotspot watchdog thread if running."""

        stop_event = self._watchdog_stop_event
        thread = self._watchdog_thread
        if stop_event is None or thread is None:
            return
        stop_event.set()
        thread.join(timeout=5.0)
        self._watchdog_thread = None
        self._watchdog_stop_event = None

    def _watchdog_cycle(self, *, initial_boot_check: bool) -> None:
        known_ssids = {
            ssid.strip()
            for ssid in self._credentials.list_networks()
            if isinstance(ssid, str) and ssid.strip()
        }
        try:
            status = self._fetch_status(update_mdns=False)
        except WiFiError as exc:
            status = None
            self._record_log(
                "hotspot_watchdog_status_error",
                f"Unable to refresh Wi-Fi status during watchdog cycle: {exc}.",
            )
        else:
            if self._status_on_known_network(status, known_ssids):
                event = "hotspot_watchdog_boot_connected" if initial_boot_check else "hotspot_watchdog_connected"
                self._record_log(
                    event,
                    "Watchdog confirms connection to a known network.",
                    status=status,
                )
                return
            if status.hotspot_active:
                self._record_log(
                    "hotspot_watchdog_hotspot_active",
                    "Watchdog detected active hotspot; skipping checks.",
                    status=status,
                )
                return

        self._record_log(
            "hotspot_watchdog_start",
            "Watchdog attempting to recover connectivity to a known network.",
            status=status,
        )
        attempts = 0
        last_status = status
        while attempts < 2:
            attempts += 1
            try:
                last_status = self.auto_connect_known_networks(
                    start_hotspot=False,
                    update_mdns=False,
                )
            except WiFiError as exc:
                self._record_log(
                    "hotspot_watchdog_connect_error",
                    f"Watchdog auto-connect attempt {attempts} failed: {exc}.",
                    status=last_status,
                )
                last_status = None
            else:
                if self._status_on_known_network(last_status, known_ssids):
                    self._record_log(
                        "hotspot_watchdog_recovered",
                        "Watchdog restored connection to a known network.",
                        status=last_status,
                    )
                    return
            if attempts < 2 and self._watchdog_stop_event is not None:
                retry_wait = self._watchdog_retry_delay
                self._record_log(
                    "hotspot_watchdog_retry",
                    f"Retrying known network connection in {retry_wait:.1f}s.",
                    status=last_status,
                )
                if retry_wait and self._watchdog_stop_event.wait(retry_wait):
                    return

        if last_status and last_status.hotspot_active:
            self._record_log(
                "hotspot_watchdog_hotspot_already_enabled",
                "Hotspot already active after auto-connect attempts.",
                status=last_status,
            )
            return

        self._record_log(
            "hotspot_watchdog_enable_hotspot",
            "Starting hotspot after watchdog auto-connect attempts failed.",
            status=last_status,
        )

        failure_attempts: list[dict[str, object | None]] = []
        try:
            first_attempt = self.enable_hotspot()
        except WiFiError as exc:
            error_metadata = self._collect_hotspot_error_context(
                {"attempt": 1, "trigger": "watchdog"}, exc
            )
            failure_attempts.append(error_metadata)
            self._record_log(
                "hotspot_watchdog_hotspot_error",
                (
                    "Unable to enable hotspot after watchdog attempts: "
                    f"{exc}."
                ),
                status=last_status,
                metadata=error_metadata if error_metadata else None,
            )
            first_attempt = None
        else:
            if first_attempt.hotspot_active:
                self._record_log(
                    "hotspot_watchdog_hotspot_enabled",
                    (
                        "Hotspot enabled by watchdog after failed reconnection "
                        "attempts."
                    ),
                    status=first_attempt,
                )
                return
            self._record_log(
                "hotspot_watchdog_hotspot_inactive",
                "Hotspot did not report as active; retrying.",
                status=first_attempt,
            )
            last_status = first_attempt

        try:
            second_attempt = self.enable_hotspot()
        except WiFiError as exc:
            error_metadata = self._collect_hotspot_error_context(
                {"attempt": 2, "trigger": "watchdog"}, exc
            )
            failure_attempts.append(error_metadata)
            self._record_log(
                "hotspot_watchdog_hotspot_error",
                (
                    "Unable to enable hotspot after watchdog attempts: "
                    f"{exc}."
                ),
                status=last_status,
                metadata=error_metadata if error_metadata else None,
            )
        else:
            if second_attempt.hotspot_active:
                self._record_log(
                    "hotspot_watchdog_hotspot_enabled",
                    (
                        "Hotspot enabled by watchdog after failed reconnection "
                        "attempts."
                    ),
                    status=second_attempt,
                )
                return
            last_status = second_attempt
            self._record_log(
                "hotspot_watchdog_hotspot_inactive",
                "Hotspot still inactive after watchdog retry.",
                status=second_attempt,
            )
        try:
            current_status = self._fetch_status(update_mdns=False)
        except WiFiError:
            current_status = None
        final_metadata: dict[str, object | None] | None = None
        if failure_attempts:
            final_metadata = {"attempts": failure_attempts}
            latest = failure_attempts[-1]
            if isinstance(latest, dict):
                for key in (
                    "diagnostics_summary",
                    "diagnostics",
                    "diagnostics_json",
                    "error_details",
                    "error_details_json",
                ):
                    value = latest.get(key) if latest else None
                    if value is not None:
                        final_metadata.setdefault(key, value)
        self._record_log(
            "hotspot_watchdog_hotspot_failed",
            "Hotspot could not be enabled after watchdog retries.",
            status=current_status,
            metadata=final_metadata,
        )

    @staticmethod
    def _status_on_known_network(status: WiFiStatus, known_ssids: set[str]) -> bool:
        if not isinstance(status, WiFiStatus):
            return False
        if not status.connected or status.hotspot_active:
            return False
        identifiers = {
            value.strip()
            for value in (status.ssid, status.profile)
            if isinstance(value, str) and value.strip()
        }
        return bool(identifiers & known_ssids)

    def _fetch_status(self, *, update_mdns: bool = True) -> WiFiStatus:
        status = self._backend.get_status()
        return self._finalise_status(status, update_mdns=update_mdns)

    def _await_station_ip(
        self,
        *,
        initial_status: WiFiStatus | None = None,
        timeout: float | None = None,
        sleep_interval: float | None = None,
    ) -> WiFiStatus | None:
        """Poll for an IP address after connecting to a network.

        Some successful joins report connectivity before DHCP assigns an
        address. Polling lets us refresh mDNS advertisements as soon as the
        lease arrives so controller devices can rediscover the camera.
        """

        if timeout is None:
            timeout = max(5.0, self._poll_interval * 5)
        if timeout <= 0:
            return initial_status
        deadline = time.monotonic() + timeout
        last_status = initial_status
        interval = sleep_interval if sleep_interval is not None else self._poll_interval
        interval = max(0.001, interval)
        while time.monotonic() < deadline:
            time.sleep(interval)
            try:
                candidate = self._fetch_status(update_mdns=True)
            except WiFiError:
                continue
            last_status = candidate
            if (
                candidate.ip_address
                or not candidate.connected
                or candidate.hotspot_active
            ):
                return candidate
        return last_status

    def _schedule_ip_refresh(
        self,
        *,
        initial_status: WiFiStatus,
        timeout: float | None = None,
    ) -> None:
        """Refresh the connection status in the background until IP assigned."""

        if timeout is None:
            timeout = 5.0
        if timeout <= 0:
            return

        def _worker() -> None:
            try:
                self._await_station_ip(
                    initial_status=initial_status,
                    timeout=timeout,
                    sleep_interval=max(self._poll_interval, 0.25),
                )
            except Exception:  # pragma: no cover - defensive thread guard
                logging.getLogger(__name__).debug(
                    "Background IP refresh failed", exc_info=True
                )

        thread = threading.Thread(
            target=_worker,
            name="wifi-ip-refresh",
            daemon=True,
        )
        thread.start()

    def _apply_hotspot_password(self, status: WiFiStatus | None) -> WiFiStatus:
        if isinstance(status, WiFiStatus):
            status.hotspot_password = self._hotspot_password
        return status

    def set_status_listener(
        self, listener: Callable[[WiFiStatus], None] | None
    ) -> None:
        """Register a callback invoked whenever the connection status updates."""

        self._status_listener = listener

    def _notify_status(self, status: WiFiStatus | None) -> None:
        listener = self._status_listener
        if listener is None or not isinstance(status, WiFiStatus):
            return
        try:
            listener(status)
        except Exception:  # pragma: no cover - defensive logging
            logging.getLogger(__name__).debug(
                "Wi-Fi status listener failed", exc_info=True
            )

    def _finalise_status(
        self,
        status: WiFiStatus | None,
        *,
        update_mdns: bool = True,
    ) -> WiFiStatus:
        result = self._apply_hotspot_password(status)
        if update_mdns:
            self._update_mdns(result)
        self._notify_status(result)
        return result

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

    def _restore_network_log(self) -> None:
        entries = self._system_log.tail(self._log.maxlen, category="network")
        if not entries:
            return
        with self._log_lock:
            for entry in entries:
                self._log.append(entry)

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
        entry = self._system_log.record(
            "network",
            event,
            message,
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

    def _collect_hotspot_error_context(
        self,
        base_metadata: dict[str, object | None] | None,
        exc: Exception,
    ) -> dict[str, object | None]:
        """Merge hotspot attempt metadata with backend diagnostics."""

        metadata = dict(base_metadata) if base_metadata else {}
        details = getattr(exc, "details", None)
        if details:
            metadata["error_details"] = details
            stringified = self._stringify_metadata(details)
            if stringified:
                metadata.setdefault("error_details_json", stringified)
        diagnostics: dict[str, object | None] | None = None
        backend_diag = getattr(self._backend, "hotspot_diagnostics", None)
        if callable(backend_diag):
            try:
                diagnostics = backend_diag()
            except Exception:  # pragma: no cover - diagnostics are best effort
                logging.getLogger(__name__).debug(
                    "Unable to collect hotspot diagnostics", exc_info=True
                )
                diagnostics = None
        if diagnostics:
            metadata.setdefault("diagnostics", diagnostics)
            diag_string = self._stringify_metadata(diagnostics)
            if diag_string:
                metadata.setdefault("diagnostics_json", diag_string)
        summary = self._summarize_hotspot_error_context(metadata)
        if summary:
            metadata.setdefault("diagnostics_summary", summary)
        return metadata

    def _summarize_hotspot_error_context(
        self, metadata: dict[str, object | None]
    ) -> str | None:
        parts: list[str] = []
        details = metadata.get("error_details")
        connection_name: str | None = None
        if isinstance(details, dict):
            raw_connection = details.get("connection")
            if isinstance(raw_connection, str) and raw_connection.strip():
                connection_name = raw_connection.strip()
                parts.append(f"connection {connection_name}")
            attempts_summary = self._summarize_hotspot_attempts(details.get("attempts"))
            if attempts_summary:
                parts.append(attempts_summary)
        diagnostics = metadata.get("diagnostics")
        diagnostics_summary = self._summarize_backend_diagnostics(diagnostics)
        if diagnostics_summary:
            parts.append(diagnostics_summary)
        if not parts:
            return None
        return "; ".join(parts)

    @staticmethod
    def _summarize_hotspot_attempts(attempts: object) -> str | None:
        if not isinstance(attempts, list):
            return None
        errors: list[str] = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            stage = attempt.get("stage")
            steps = attempt.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("result") != "error":
                    continue
                action = step.get("action")
                property_name = step.get("property")
                descriptor_parts: list[str] = []
                if isinstance(stage, str) and stage.strip():
                    descriptor_parts.append(stage.strip())
                if isinstance(action, str) and action.strip():
                    descriptor_parts.append(action.strip())
                if isinstance(property_name, str) and property_name.strip():
                    descriptor_parts.append(property_name.strip())
                descriptor = " ".join(descriptor_parts) if descriptor_parts else "step"
                error_message = step.get("error")
                if not isinstance(error_message, str) or not error_message.strip():
                    error_message = "error"
                else:
                    error_message = error_message.strip()
                errors.append(f"{descriptor} failed: {error_message}")
        if not errors:
            return None
        if len(errors) > 3:
            remaining = len(errors) - 3
            return ", ".join(errors[:3]) + f", and {remaining} more"
        return ", ".join(errors)

    @staticmethod
    def _summarize_backend_diagnostics(diagnostics: object) -> str | None:
        if not isinstance(diagnostics, dict):
            return None
        pieces: list[str] = []
        state = diagnostics.get("state")
        if isinstance(state, str) and state.strip():
            pieces.append(f"state={state.strip()}")
        error_value = diagnostics.get("error")
        if isinstance(error_value, str) and error_value.strip():
            pieces.append(f"backend error: {error_value.strip()}")
        requires_secret = diagnostics.get("requires_secret")
        if isinstance(requires_secret, bool):
            if requires_secret:
                pieces.append("backend reports hotspot still expects secrets")
            else:
                pieces.append("backend reports hotspot secrets cleared")
        connection = diagnostics.get("connection")
        if isinstance(connection, str) and connection.strip():
            pieces.append(f"diagnostics connection {connection.strip()}")
        if not pieces:
            return None
        return ", ".join(pieces)

    @staticmethod
    def _stringify_metadata(value: object) -> str | None:
        try:
            return json.dumps(value, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return None


__all__ = [
    "WiFiCredentialStore",
    "WiFiError",
    "WiFiNetwork",
    "WiFiStatus",
    "WiFiBackend",
    "NMCLIBackend",
    "WiFiManager",
]
