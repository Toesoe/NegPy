import threading
from collections.abc import Iterable, Iterator
from typing import Callable

import numpy as np

from negpy.infrastructure.scanners.base import (
    ScanMode,
    ScannerCapabilities,
    ScannerDevice,
    ScannerUnavailable,
    TransientScanError,
)

from negpy.infrastructure.scanners.roll import (
    RollPreview,
    RollSession,
)

from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_PREPARE_WAIT_FOR_MEDIA_S = 300.0

# What an SA-21 strip carrier holds, used when the transport will not say. Only a bound: the
# user narrows it with the frame range.
_STRIP_FRAMES = 6


def _caps_from_nkscan(caps, sensed_frames: int | None = None) -> ScannerCapabilities:
    """Build ScannerCapabilities from an nkscan.Capabilities. Pure — no `nkscan` import.

    `sensed_frames` is what the loaded carrier reports, when it was readable.
    """
    return ScannerCapabilities(
        ir_channel=bool(caps.ir_channel),
        supported_dpi=tuple(sorted(int(d) for d in caps.dpi)),
        supported_depths=tuple([caps.depth,]),
        # nkscan devices are dedicated film scanners with no source selector at all.
        sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE, ScanMode.TRANSPARENCY),
        max_area_mm=(float(caps.max_area_mm[0]), float(caps.max_area_mm[1])),
        auto_exposure=bool(caps.auto_exposure),
        adapter_frame_capacity=_strip_capacity(sensed_frames),
        can_eject=bool(caps.can_eject),
        multisample=tuple(sorted(int(m) for m in caps.multisample)) or (1,),
        single_line=bool(caps.single_line),
        supports_exposure_lock=True,  # every nkscan session can lock_gain()
        # The white balance lock rides on host-side metering; a model that meters in firmware
        # takes the request and ignores it.
        supports_white_balance_lock=bool(caps.auto_exposure),
    )


def _strip_capacity(sensed_frames: int | None) -> int:
    """Frame positions to offer for a loaded carrier.

    A sensed count of exactly 1 is ambiguous — the transport reports 1 both for a single frame
    and for any strip that has already been passed once — so it falls back to the carrier bound
    rather than collapsing the frame range after a scan.
    """
    if sensed_frames is not None and sensed_frames > 1:
        return sensed_frames
    return _STRIP_FRAMES


def _as_scan_error(exc: Exception) -> Exception:
    """Re-type an nkscan transient failure so the service can retry without reading messages."""
    import nkscan

    if isinstance(exc, nkscan.TransientError):
        return TransientScanError(str(exc))
    return exc


def _progress_adapter(
    progress: Callable[[float], None] | None,
    cancel: threading.Event,
) -> Callable[[int, int], bool]:
    """Adapt nkscan's `(read, total) -> bool | None` mid-read callback (returning False
    aborts the pass) to NegPy's fractional `progress(float)` callable plus the shared
    cancel Event, giving real mid-scan cancellation."""

    def _cb(read: int, total: int) -> bool:
        if progress is not None and total:
            try:
                progress(min(1.0, max(0.0, read / total)))
            except Exception:
                pass
        return not cancel.is_set()

    return _cb


class NkscanSession:
    """Wraps one nkscan.Session: lazily prepares on first scan(), then reuses the
    placed session for subsequent frames without re-preparing."""

    def __init__(self, session, *, lock_white_balance: bool = False) -> None:
        self._session = session
        self._prepared = False
        self.device_id: str = session.device_id
        # Held during frame 1's own auto-exposure metering — independent of
        # lock_exposure() (which freezes the *gain* a metered frame settled on).
        # WB lock without a gain freeze still re-meters brightness each frame,
        # just without neutralizing the film's orange base while doing it.
        self._lock_white_balance = lock_white_balance

    def _ensure_prepared(self, params: ScanParams) -> None:
        if self._prepared:
            return
        try:
            self._session.prepare(
                # One prepare covers the whole strip, so the table has to reach the furthest
                # frame the caller will address or everything past the first comes back black.
                frames=params.frame_count or params.frame,
                pitch_mm=None,
                offset_mm=params.frame_offset_mm,
                offsets_mm=list(params.frame_offsets_mm) if params.frame_offsets_mm else None,
                gain=None,  # auto-exposure; a caller freezes it afterwards via lock_exposure()
                lock_white_balance=self._lock_white_balance,
                wait_for_media_s=_PREPARE_WAIT_FOR_MEDIA_S,
            )
        except Exception as e:
            raise _as_scan_error(e) from e
        self._prepared = True

    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        if cancel.is_set():
            # A cancel set before this call must not pay for prepare()'s hardware
            # wait (up to 300s for media) or a real scan.
            raise RuntimeError("Scan cancelled before start")
        self._ensure_prepared(params)
        # NegPy's ScanParams.frame is 1-indexed (batch ranges are `range(frame_from, frame_to+1)`);
        # nkscan's scan(index=...) is explicitly 0-indexed. Deliberate translation, not an oversight.
        index = (params.frame - 1) if params.frame is not None else 0
        try:
            result = self._session.scan(
                index,
                dpi=params.dpi,
                ir=params.capture_ir,
                # nkscan has no way to report back or accept a prior autofocus setpoint, so
                # ScanParams.autofocus=False cannot be honoured — always autofocus fresh.
                focus="auto",
                multisample=params.multisample,
                single_line=params.single_line,
                window=params.window,
                progress=_progress_adapter(progress, cancel),
            )
        except Exception as e:
            raise _as_scan_error(e) from e
        # nkscan's IR plane always covers the full frame.
        ir_valid_mask = np.ones(result.ir.shape[:2], dtype=np.bool_) if result.ir is not None else None
        return ScanResult(
            rgb=result.rgb,
            ir=result.ir,
            dpi=result.dpi,
            device_model=result.device_model,
            ir_valid_mask=ir_valid_mask,
        )

    def lock_exposure(self) -> None:
        self._session.lock_gain()

    def eject(self) -> bool:
        # Capability-gated no-op for a device with no eject action, matching
        # SaneBackend.eject()'s contract — nkscan's own eject() has no such gate.
        if not bool(self._session.capabilities.can_eject):
            self.close()
            return False
        try:
            return bool(self._session.eject())
        except Exception as e:
            raise _as_scan_error(e) from e
        finally:
            self.close()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "NkscanSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class NkscanBackend:
    """nkscan implementation of ScannerBackend. Only module that imports `nkscan`."""

    def __init__(self) -> None:
        try:
            import nkscan  # noqa: F401
        except ImportError:
            raise ScannerUnavailable("nkscan not importable. Install: uv sync --group nkscan") from None
        self._nkscan = nkscan
        self._devices_cache: list[ScannerDevice] | None = None

    def list_devices(self) -> list[ScannerDevice]:
        if self._devices_cache is not None:
            return self._devices_cache
        try:
            raw_devices = self._nkscan.list_devices()
        except Exception as e:
            logger.error(f"nkscan device listing failed: {e}")
            return []
        devices = [ScannerDevice(id=d.id, vendor=d.vendor, model=d.model, capabilities=self._probe(d)) for d in raw_devices]
        self._devices_cache = devices
        return devices

    def _probe(self, device) -> ScannerCapabilities:
        """Capabilities for one enumerated device, refined by the device's own answer.

        `list_devices` reports a static per-model table — enumeration must not reserve a
        scanner — so the film area it gives is the format rather than the loaded adapter's
        boundaries (36.8 mm against the 37.84 mm an LS-50 with an SA-21 reports). A session
        answers that and the sensed frame count from vendor pages at no mechanical cost, so
        one open settles both. It is not always available: a device held by a running scan
        keeps the enumeration table and the carrier bound.
        """
        try:
            session = self._nkscan.Session(device.id)
        except Exception as e:
            logger.debug(f"Could not open {device.id} for its own capabilities: {e}")
            return _caps_from_nkscan(device.capabilities)
        try:
            return _caps_from_nkscan(session.capabilities, session.sensed_frames())
        finally:
            session.close()

    def refresh_devices(self) -> list[ScannerDevice]:
        self._devices_cache = None
        return self.list_devices()

    def open_session(self, device_id: str, *, lock_white_balance: bool = False) -> NkscanSession:
        try:
            session = self._nkscan.Session(device_id)
        except Exception as e:
            raise _as_scan_error(e) from e
        return NkscanSession(session, lock_white_balance=lock_white_balance)
    
    def open_roll(self, device_id: str, *, dpi: int):
        session = self._nkscan.Session(device_id)
        return NkscanRollSession(session, dpi=dpi)

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        with self.open_session(device_id) as session:
            return session.scan(params, progress, cancel)
        
    def roll_preview(
        self,
        device_id: str,
        cancel: threading.Event,
    ) -> np.ndarray:
        """Preview the whole strip in one pass, returning a single RGB array.

        The caller can slice it into frames and apply offsets. The transport may
        not support this, in which case it raises ScannerUnavailable.
        """
        with self.open_session(device_id) as session:
            try:
                return session.roll_preview(progress=lambda _fraction: None, cancel=cancel).rgb
            except Exception as e:
                raise _as_scan_error(e) from e

    def eject(self, device_id: str) -> bool:
        # NkscanSession.eject() already terminates the session itself; no `with`
        # here, or it would close a second time (harmless — Session.close() is
        # idempotent — but pointless).
        return self.open_session(device_id).eject()

class NkscanRollSession:
    """
    Whole-strip preview using nkscan's native roll preview traversal.

    The scanner moves once, returning a complete strip raster. Frame boundaries
    are projected from the adapter pitch and returned as individual previews.
    """

    def __init__(self, session, *, dpi: int):
        self._session = session
        self._dpi = dpi

        caps = session.capabilities

        self.slot_count = session.sensed_frames() or 1

        self.offset_range = (0.0, 1.0)

        # Native traversal means scanning one slot separately is meaningless.
        self.supports_single_slot_preview = False

        self._pitch_mm = 25.4 / dpi

        self._offsets: dict[int, float] = {}

        self._closed = False


    def set_offset(self, slot: int, offset: float) -> None:
        self._offsets[slot] = offset


    def approve(self, slot: int) -> None:
        # Native preview does not infer boundaries yet.
        pass


    def preview(
        self,
        slots: Iterable[int],
        *,
        cancel: threading.Event,
    ) -> Iterator[RollPreview]:

        if cancel.is_set():
            return

        try:
            strip = self._session.overview(
                progress=lambda *_: not cancel.is_set(),
            )

        except Exception as e:
            logger.warning("Whole-strip preview failed: %s", e)
            for slot in slots:
                yield RollPreview(
                    slot=slot,
                    error=str(e),
                )
            return


        frames = self._split_frames(strip.rgb)

        wanted = set(slots)

        for index, frame in enumerate(frames, start=1):
            if cancel.is_set():
                return

            if index not in wanted:
                continue

            yield RollPreview(
                slot=index,
                rgb=frame,
                offset=self._offset_fraction(index),
            )


    def _split_frames(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Split the continuous strip raster into frame previews.

        Currently assumes equal pitch. Later this can use boundary detection.
        """

        if self.slot_count <= 1:
            return [image]

        height = image.shape[0]

        frame_height = height / self.slot_count

        frames = []

        for i in range(self.slot_count):
            start = round(i * frame_height)
            end = round((i + 1) * frame_height)

            frames.append(
                image[start:end, :]
            )

        return frames


    def _offset_fraction(self, slot: int) -> float:
        return self._offsets.get(slot, 0.0)


    def close(self) -> None:
        if not self._closed:
            self._session.close()
            self._closed = True