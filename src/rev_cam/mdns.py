"""Simple mDNS advertiser used to expose RevCam services."""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import shutil
import subprocess
import threading
import time
from typing import Callable, TypeVar, cast

try:  # pragma: no cover - import guard for optional dependency failures
    from zeroconf import InterfaceChoice, ServiceInfo, Zeroconf
except Exception as exc:  # pragma: no cover - dependency import failure
    Zeroconf = None  # type: ignore[assignment]
    ServiceInfo = None  # type: ignore[assignment]
    InterfaceChoice = None  # type: ignore[assignment]
    _zeroconf_error = exc
else:
    _zeroconf_error = None


logger = logging.getLogger(__name__)


_T = TypeVar("_T")


class _BaseAdvertiser:
    """Internal protocol for advertiser backends."""

    def advertise(self, ip_address: str | None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def clear(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _ZeroconfAdvertiser(_BaseAdvertiser):
    """Advertise ``motion.local`` using the zeroconf Python package."""

    def __init__(
        self,
        hostname: str,
        service_name: str,
        service_type: str,
        port: int,
    ) -> None:
        if Zeroconf is None or ServiceInfo is None or InterfaceChoice is None:
            reason = _zeroconf_error or "zeroconf library unavailable"
            raise RuntimeError(reason)
        self._hostname = hostname
        self._service_name = service_name
        self._service_type = service_type
        self._port = port
        self._zeroconf: Zeroconf | None = None
        self._info: ServiceInfo | None = None
        self._current_ip: str | None = None
        self._disabled_reason: str | None = None

    # ------------------------------ helpers -----------------------------
    def _ensure_zeroconf(self) -> Zeroconf:
        if self._zeroconf is None:
            self._zeroconf = self._create_zeroconf()
        return self._zeroconf

    def _create_zeroconf(self) -> Zeroconf:
        return self._run_blocking(lambda: Zeroconf(interfaces=InterfaceChoice.All))

    @staticmethod
    def _event_loop_running() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _run_blocking(self, func: Callable[[], _T]) -> _T:
        if not self._event_loop_running():
            return func()

        outcome: object | None = None
        error: BaseException | None = None
        finished = threading.Event()

        def _invoke() -> None:
            nonlocal outcome, error
            try:
                outcome = func()
            except BaseException as exc:  # pragma: no cover - defensive guard
                error = exc
            finally:
                finished.set()

        thread = threading.Thread(target=_invoke, daemon=True)
        thread.start()
        finished.wait()

        if error is not None:
            raise error
        return cast(_T, outcome)

    def _build_service_info(self, ip: ipaddress._BaseAddress) -> ServiceInfo:
        service_name = f"{self._service_name}.{self._service_type}"
        if not service_name.endswith("."):
            service_name = f"{service_name}."
        server = f"{self._hostname}."
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
        if self._disabled_reason is not None:
            return
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
                self._run_blocking(lambda: zeroconf.unregister_service(self._info))
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.debug("Ignoring mDNS unregister failure: %s", exc)
            self._info = None
        info = self._build_service_info(parsed_ip)
        try:
            self._run_blocking(
                lambda: zeroconf.register_service(info, allow_name_change=False)
            )
        except Exception as exc:  # pragma: no cover - zeroconf runtime issues
            message = (str(exc) or exc.__class__.__name__).strip()
            lowered = message.lower()
            permission_related = (
                isinstance(exc, PermissionError)
                or "permission" in lowered
                or "sudo" in lowered
                or "/dev/mem" in lowered
            )
            if permission_related and self._disabled_reason is None:
                if "/dev/mem" in lowered or "sudo" in lowered:
                    logger.warning(
                        "mDNS advertising disabled: insufficient privileges to access "
                        "the NeoPixel driver (%s). Run RevCam with sudo or configure "
                        "an alternate LED backend to restore lighting and mDNS.",
                        message,
                    )
                else:
                    logger.warning(
                        "mDNS advertising disabled due to insufficient privileges: %s", message
                    )
                self._disabled_reason = message
                self._info = None
                self._current_ip = None
                if self._zeroconf is not None:
                    try:
                        self._run_blocking(self._zeroconf.close)
                    except Exception:  # pragma: no cover - best effort cleanup
                        logger.debug(
                            "Ignoring Zeroconf close failure after permission error",
                            exc_info=True,
                        )
                    finally:
                        self._zeroconf = None
                return
            logger.warning("Failed to register mDNS service: %s", message)
            return
        self._info = info
        self._current_ip = ip_address

    def clear(self) -> None:
        if self._info is None:
            self._current_ip = None
            return
        zeroconf = self._zeroconf
        if zeroconf is not None:
            try:
                self._run_blocking(lambda: zeroconf.unregister_service(self._info))
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.debug("Ignoring mDNS unregister failure: %s", exc)
        self._info = None
        self._current_ip = None

    def close(self) -> None:
        self.clear()
        if self._zeroconf is not None:
            try:
                self._run_blocking(self._zeroconf.close)
            finally:
                self._zeroconf = None


class _AvahiAdvertiser(_BaseAdvertiser):
    """Advertise using the ``avahi-publish`` CLI as a lightweight fallback."""

    def __init__(
        self,
        hostname: str,
        service_name: str,
        service_type: str,
        port: int,
    ) -> None:
        del service_name, service_type, port  # Service metadata is unused for address records.
        binary = shutil.which("avahi-publish")
        if not binary:
            raise FileNotFoundError("avahi-publish command not found")
        self._binary = binary
        self._hostname = hostname
        self._process: subprocess.Popen[str] | None = None
        self._current_ip: str | None = None
        self._lock = threading.Lock()

    def advertise(self, ip_address: str | None) -> None:
        with self._lock:
            if ip_address is None:
                self._stop_process()
                return
            if self._process is not None and self._process.poll() is not None:
                self._drain_process(self._process, unexpected=True)
                self._process = None
                self._current_ip = None
            if ip_address == self._current_ip and self._process is not None:
                return
            self._stop_process()
            self._start_process(ip_address)

    def clear(self) -> None:
        with self._lock:
            self._stop_process()

    def close(self) -> None:
        self.clear()

    # ------------------------------ helpers -----------------------------
    def _start_process(self, ip_address: str) -> None:
        command = [self._binary, "-a", "-R", self._hostname, ip_address]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(f"failed to launch avahi-publish: {exc}") from exc
        # avahi-publish exits immediately when it cannot register the record.
        time.sleep(0.1)
        if process.poll() is not None:
            message = self._read_process_error(process)
            raise RuntimeError(
                f"avahi-publish exited with code {process.returncode}: {message}"
            )
        self._process = process
        self._current_ip = ip_address

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            self._current_ip = None
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        self._drain_process(process, unexpected=False)
        self._process = None
        self._current_ip = None

    def _drain_process(self, process: subprocess.Popen[str], *, unexpected: bool) -> None:
        message = self._read_process_error(process)
        if message:
            if unexpected:
                logger.warning("avahi-publish exited unexpectedly: %s", message)
            else:
                logger.debug("avahi-publish output: %s", message)
        if process.stderr:
            try:
                process.stderr.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    @staticmethod
    def _read_process_error(process: subprocess.Popen[str]) -> str:
        if process.stderr is None:
            return ""
        try:
            data = process.stderr.read()
        except Exception:  # pragma: no cover - best effort cleanup
            return ""
        return data.strip()


class _NullAdvertiser(_BaseAdvertiser):
    """No-op advertiser used when no backend is available."""

    def advertise(self, ip_address: str | None) -> None:  # pragma: no cover - no behaviour
        del ip_address

    def clear(self) -> None:  # pragma: no cover - no behaviour
        return

    def close(self) -> None:  # pragma: no cover - no behaviour
        return


class MDNSAdvertiser(_BaseAdvertiser):
    """Manage an mDNS advertisement for the RevCam HTTP service."""

    def __init__(
        self,
        hostname: str = "motion.local",
        *,
        service_name: str = "RevCam",
        service_type: str = "_http._tcp.local.",
        port: int = 9000,
    ) -> None:
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
        self._backend = self._select_backend()

    # ------------------------------ helpers -----------------------------
    def _select_backend(self) -> _BaseAdvertiser:
        errors: list[str] = []
        if Zeroconf is not None and ServiceInfo is not None and InterfaceChoice is not None:
            try:
                return _ZeroconfAdvertiser(
                    self._hostname,
                    self._service_name,
                    self._service_type,
                    self._port,
                )
            except Exception as exc:  # pragma: no cover - rare runtime error
                errors.append(f"zeroconf backend failed: {exc}")
        elif _zeroconf_error is not None:
            errors.append(f"zeroconf import failed: {_zeroconf_error}")
        try:
            return _AvahiAdvertiser(
                self._hostname,
                self._service_name,
                self._service_type,
                self._port,
            )
        except FileNotFoundError:
            errors.append("avahi-publish utility not available")
        except Exception as exc:  # pragma: no cover - rare runtime error
            errors.append(f"avahi backend failed: {exc}")
        if errors:
            logger.warning(
                "mDNS advertising disabled: %s. Install the 'zeroconf' Python package or "
                "the 'avahi-utils' tools to enable motion.local announcements.",
                "; ".join(errors),
            )
        else:  # pragma: no cover - defensive
            logger.warning(
                "mDNS advertising disabled: no backend available. Install the 'zeroconf' "
                "package or the 'avahi-utils' tools to enable motion.local announcements.",
            )
        return _NullAdvertiser()

    # ------------------------------ control ----------------------------
    def advertise(self, ip_address: str | None) -> None:
        self._backend.advertise(ip_address)

    def clear(self) -> None:
        self._backend.clear()

    def close(self) -> None:
        self._backend.close()


__all__ = ["MDNSAdvertiser"]
