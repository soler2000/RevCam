"""Wi-Fi management utilities with safe fallbacks for RevCam."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Deque, Iterable, Sequence

from .mdns import MDNSAdvertiser


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
        password_provided = bool(password)
        if password_provided:
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
                "password",
                password,
            ]
            self._run(args)
        else:
            # Reconfigure the hotspot connection explicitly when no password is
            # requested so NetworkManager does not retain an old passphrase.
            connection_exists = True
            try:
                self._run(["nmcli", "connection", "show", connection_name])
            except WiFiError:
                connection_exists = False
            if not connection_exists:
                try:
                    self._run(
                        [
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
                    )
                except WiFiError as exc:
                    raise WiFiError(
                        f"Unable to prepare hotspot connection: {exc}"
                    ) from exc
            else:
                try:
                    self._run(
                        [
                            "nmcli",
                            "connection",
                            "modify",
                            connection_name,
                            "connection.interface-name",
                            interface,
                        ]
                    )
                    self._run(
                        [
                            "nmcli",
                            "connection",
                            "modify",
                            connection_name,
                            "wifi.ssid",
                            ssid,
                        ]
                    )
                except WiFiError as exc:
                    raise WiFiError(
                        f"Unable to refresh hotspot configuration: {exc}"
                    ) from exc
            try:
                base_configuration = [
                    [
                        "nmcli",
                        "connection",
                        "modify",
                        connection_name,
                        "802-11-wireless.mode",
                        "ap",
                    ],
                    [
                        "nmcli",
                        "connection",
                        "modify",
                        connection_name,
                        "ipv4.method",
                        "shared",
                    ],
                    [
                        "nmcli",
                        "connection",
                        "modify",
                        connection_name,
                        "ipv6.method",
                        "ignore",
                    ],
                    [
                        "nmcli",
                        "connection",
                        "modify",
                        connection_name,
                        "connection.autoconnect",
                        "yes",
                    ],
                ]
                for command in base_configuration:
                    self._run(command)
                security_cleared = self._reset_hotspot_security(connection_name)
                if not security_cleared:
                    self._recreate_hotspot_connection(
                        interface, connection_name, ssid
                    )
                    for command in base_configuration:
                        self._run(command)
                    if not self._reset_hotspot_security(connection_name):
                        raise WiFiError(
                            "Unable to clear hotspot security configuration"
                        )
                try:
                    self._run(["nmcli", "connection", "down", connection_name])
                except WiFiError:
                    # Connection might not be active yet; ignore and proceed.
                    pass
                self._run(["nmcli", "connection", "up", connection_name])
            except WiFiError as exc:
                raise WiFiError(f"Unable to configure open hotspot: {exc}") from exc
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
    ) -> bool:
        """Remove a security property for the given connection.

        Returns True if NetworkManager accepted either the removal or provided
        fallback assignment, False otherwise.
        """

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
            return True
        except WiFiError:
            pass
        if fallback_value is None:
            return False
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
            return True
        except WiFiError:
            return False

    def _connection_security_requires_secret(self, connection_name: str) -> bool:
        """Return True when the connection still expects a Wi-Fi secret."""

        try:
            output = self._run(
                [
                    "nmcli",
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
        except WiFiError:
            return False
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

    def _reset_hotspot_security(self, connection_name: str) -> bool:
        """Ensure the hotspot connection no longer expects a secret."""

        logger = logging.getLogger(__name__)
        removed = self._remove_security_setting(connection_name)
        cleared_any = False
        for property_name, fallback in (
            ("802-11-wireless-security.auth-alg", "open"),
            ("802-11-wireless-security.psk", None),
            ("802-11-wireless-security.psk-flags", "0"),
            ("802-11-wireless-security.wep-key0", None),
            ("802-11-wireless-security.wep-key1", None),
            ("802-11-wireless-security.wep-key2", None),
            ("802-11-wireless-security.wep-key3", None),
            ("802-11-wireless-security.wep-key-flags", "0"),
            ("802-11-wireless-security.wep-key-type", None),
            ("802-11-wireless-security.wep-tx-keyidx", "0"),
        ):
            if self._clear_security_property(
                connection_name, property_name, fallback_value=fallback
            ):
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
            success = True
        except WiFiError as exc:
            if not success:
                raise WiFiError(
                    f"Unable to clear hotspot security configuration: {exc}"
                ) from exc
            logger.debug(
                "Unable to update hotspot key management to none: %s", exc
            )
        if self._connection_security_requires_secret(connection_name):
            logger.debug(
                "Hotspot connection %s still references secrets after reset",
                connection_name,
            )
            return False
        return True

    def _remove_security_setting(self, connection_name: str) -> bool:
        """Remove the entire security setting from the connection."""

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
            return True
        except WiFiError:
            return False

    def _recreate_hotspot_connection(
        self, interface: str, connection_name: str, ssid: str
    ) -> None:
        """Delete and recreate the hotspot profile to drop lingering secrets."""

        try:
            self._run(["nmcli", "connection", "delete", connection_name])
        except WiFiError:
            # Connection may not exist yet; proceed with creation regardless.
            pass
        self._run(
            [
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
        )

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
        credential_store: WiFiCredentialStore | None = None,
        watchdog_boot_delay: float = 30.0,
        watchdog_interval: float = 30.0,
        watchdog_retry_delay: float = 5.0,
        log_path: Path | str | None = Path("data/wifi_log.jsonl"),
    ) -> None:
        self._backend = backend or NMCLIBackend()
        self._rollback_timeout = rollback_timeout
        self._poll_interval = max(0.1, poll_interval)
        self._hotspot_profile: str | None = None
        self._credentials = credential_store or WiFiCredentialStore()
        self._hotspot_password: str | None = self._credentials.get_hotspot_password()
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
        self._log: Deque[WiFiLogEntry] = deque(maxlen=200)
        self._log_lock = threading.Lock()
        self._log_path: Path | None
        if log_path is None:
            self._log_path = None
        else:
            path = Path(log_path)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover - filesystem errors rare
                logging.getLogger(__name__).warning("Unable to prepare Wi-Fi log directory: %s", exc)
                self._log_path = None
            else:
                self._log_path = path
                self._load_persistent_log()
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop_event: threading.Event | None = None
        self._watchdog_boot_delay = max(0.0, watchdog_boot_delay)
        self._watchdog_interval = max(0.0, watchdog_interval)
        self._watchdog_retry_delay = max(0.0, watchdog_retry_delay)
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
        self._backend.forget(identifier)
        if identifier == self._hotspot_profile:
            self._hotspot_profile = None
        self._credentials.forget_network(identifier)
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
        status = self._apply_hotspot_password(status)
        self._update_mdns(status)
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
            current = self._fetch_status(update_mdns=False)
        restored = self._backend.activate_profile(previous_profile)
        restored = self._apply_hotspot_password(restored)
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
        if password is not None and password.strip() and len(password.strip()) < 8:
            raise WiFiError("Hotspot password must be at least 8 characters")
        cleaned_password = password.strip() if isinstance(password, str) and password.strip() else None
        attempt_metadata: dict[str, object | None] = {
            "target": cleaned_ssid,
            "development_mode": development_mode,
            "password_provided": cleaned_password is not None,
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
            self._record_log(
                "hotspot_enable_error",
                error_message,
                metadata=attempt_metadata,
            )
            raise WiFiError(f"Unable to enable hotspot: {message}") from exc
        self._hotspot_password = cleaned_password
        self._credentials.set_hotspot_password(cleaned_password)
        status = self._apply_hotspot_password(status)
        self._hotspot_profile = status.profile or self._hotspot_profile
        self._update_mdns(status)
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
            restored = self._apply_hotspot_password(restored)
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
        current = self._apply_hotspot_password(current)
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
        status = self._apply_hotspot_password(status)
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
                first_cycle = False

        thread = threading.Thread(target=_run, name="wifi-hotspot-watchdog", daemon=True)
        self._watchdog_thread = thread
        thread.start()

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
        retry_delay = self._watchdog_retry_delay

        try:
            first_attempt = self.enable_hotspot()
        except WiFiError as exc:
            self._record_log(
                "hotspot_watchdog_hotspot_error",
                (
                    "Unable to enable hotspot after watchdog attempts: "
                    f"{exc}."
                ),
                status=last_status,
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
            self._record_log(
                "hotspot_watchdog_hotspot_error",
                (
                    "Unable to enable hotspot after watchdog attempts: "
                    f"{exc}."
                ),
                status=last_status,
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
        self._record_log(
            "hotspot_watchdog_hotspot_failed",
            "Hotspot could not be enabled after watchdog retries.",
            status=current_status,
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
        status = self._apply_hotspot_password(status)
        if update_mdns:
            self._update_mdns(status)
        return status

    def _apply_hotspot_password(self, status: WiFiStatus | None) -> WiFiStatus:
        if isinstance(status, WiFiStatus):
            status.hotspot_password = self._hotspot_password
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
            self._append_persistent_log(entry)
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

    def _load_persistent_log(self) -> None:
        if self._log_path is None or not self._log_path.exists():
            return
        try:
            with self._log_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:  # pragma: no cover - best effort logging
            logging.getLogger(__name__).warning("Unable to load Wi-Fi event log: %s", exc)
            return
        restored: Deque[WiFiLogEntry] = deque(maxlen=self._log.maxlen)
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            entry = self._deserialize_log_entry(payload)
            if entry is not None:
                restored.append(entry)
        if restored:
            with self._log_lock:
                for entry in restored:
                    self._log.append(entry)

    def _deserialize_log_entry(self, payload: object) -> WiFiLogEntry | None:
        if not isinstance(payload, dict):
            return None
        event = payload.get("event")
        message = payload.get("message")
        timestamp = payload.get("timestamp")
        if not isinstance(event, str) or not isinstance(message, str):
            return None
        try:
            ts_value = float(timestamp) if timestamp is not None else time.time()
        except (TypeError, ValueError):
            ts_value = time.time()
        status_payload = payload.get("status")
        if not isinstance(status_payload, dict):
            status_payload = None
        metadata_payload = payload.get("metadata")
        if not isinstance(metadata_payload, dict):
            metadata_payload = None
        return WiFiLogEntry(
            timestamp=ts_value,
            event=event,
            message=message,
            status=status_payload,
            metadata=metadata_payload,
        )

    def _append_persistent_log(self, entry: WiFiLogEntry) -> None:
        if self._log_path is None:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
        except OSError as exc:  # pragma: no cover - best effort logging
            logging.getLogger(__name__).warning("Unable to persist Wi-Fi event log: %s", exc)


__all__ = [
    "WiFiError",
    "WiFiNetwork",
    "WiFiStatus",
    "WiFiBackend",
    "NMCLIBackend",
    "WiFiManager",
    "WiFiLogEntry",
]
