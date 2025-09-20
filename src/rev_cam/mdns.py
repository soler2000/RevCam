"""Simple mDNS advertiser used to expose RevCam services."""
from __future__ import annotations

import ipaddress
import logging

try:  # pragma: no cover - import guard for optional dependency failures
    from zeroconf import InterfaceChoice, ServiceInfo, Zeroconf
except Exception as exc:  # pragma: no cover - dependency import failure
    Zeroconf = None  # type: ignore[assignment]
    ServiceInfo = None  # type: ignore[assignment]
    InterfaceChoice = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None


logger = logging.getLogger(__name__)


class MDNSAdvertiser:
    """Manage an mDNS advertisement for the RevCam HTTP service."""

    def __init__(
        self,
        hostname: str = "motion.local",
        *,
        service_name: str = "RevCam",
        service_type: str = "_http._tcp.local.",
        port: int = 9000,
    ) -> None:
        if _import_error is not None:  # pragma: no cover - environment specific
            raise RuntimeError(f"zeroconf library unavailable: {_import_error}")
        cleaned = hostname.strip().rstrip(".")
        if not cleaned:
            raise ValueError("mDNS hostname must be provided")
        if not cleaned.endswith(".local"):
            cleaned = f"{cleaned}.local"
        if not service_type.endswith("."):
            service_type = f"{service_type}."
        self._hostname = cleaned
        self._service_name = service_name.strip() or "RevCam"
        self._service_type = service_type
        self._port = int(port)
        self._zeroconf: Zeroconf | None = None
        self._info: ServiceInfo | None = None
        self._current_ip: str | None = None

    # ------------------------------ helpers -----------------------------
    def _ensure_zeroconf(self) -> Zeroconf:
        if self._zeroconf is None:
            assert InterfaceChoice is not None  # for type checkers
            self._zeroconf = Zeroconf(interfaces=InterfaceChoice.All)
        return self._zeroconf

    def _build_service_info(self, ip: ipaddress._BaseAddress) -> ServiceInfo:
        service_name = f"{self._service_name}.{self._service_type}"
        if not service_name.endswith("."):
            service_name = f"{service_name}."
        server = f"{self._hostname}."
        assert ServiceInfo is not None  # for type checkers
        return ServiceInfo(
            type_=self._service_type,
            name=service_name,
            addresses=[ip.packed],
            port=self._port,
            server=server,
            properties={"path": b"/"},
        )

    # ------------------------------ control ----------------------------
    def advertise(self, ip_address: str | None) -> None:
        """Publish an A/AAAA record for ``hostname`` at ``ip_address``."""

        if ip_address is None:
            self.clear()
            return
        if ip_address == self._current_ip:
            return
        try:
            parsed_ip = ipaddress.ip_address(ip_address)
        except ValueError:
            logger.warning("Ignoring invalid IP address for mDNS: %s", ip_address)
            return
        try:
            zeroconf = self._ensure_zeroconf()
        except OSError as exc:  # pragma: no cover - environment specific
            logger.warning("Unable to start mDNS announcer: %s", exc)
            return
        if self._info is not None:
            try:
                zeroconf.unregister_service(self._info)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.debug("Ignoring mDNS unregister failure: %s", exc)
            self._info = None
        info = self._build_service_info(parsed_ip)
        try:
            zeroconf.register_service(info, allow_name_change=False)
        except Exception as exc:  # pragma: no cover - zeroconf runtime issues
            logger.warning("Failed to register mDNS service: %s", exc)
            return
        self._info = info
        self._current_ip = ip_address

    def clear(self) -> None:
        """Withdraw the current advertisement."""

        if self._info is None:
            self._current_ip = None
            return
        zeroconf = self._zeroconf
        if zeroconf is not None:
            try:
                zeroconf.unregister_service(self._info)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.debug("Ignoring mDNS unregister failure: %s", exc)
        self._info = None
        self._current_ip = None

    def close(self) -> None:
        """Stop advertising and release the zeroconf socket."""

        self.clear()
        if self._zeroconf is not None:
            try:
                self._zeroconf.close()
            finally:
                self._zeroconf = None


__all__ = ["MDNSAdvertiser"]
