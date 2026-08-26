"""psix backend — Kodak Pakon F135/F135+ via the psix web app (HTTP).

psix is a local web app that drives the Pakon over USB: firmware load, whole-strip
transport, and a C41 develop pipeline. This backend talks to its HTTP API
(default http://127.0.0.1:5135, override with the PSIX_URL env var).

One strip scan = one full film traversal inside psix. psix develops the capture
into raw negatives — uninverted, orange mask intact, frame-detected and sliced —
and applies ICE (infrared dust/scratch removal) when the scan was 4-channel.
Those developed frames are enumerated and handed over as ScanResults; no preview
exists because the transport cannot stop mid-strip.

Acquisition mode: whole-strip (capabilities.whole_strip). The sidebar starts one
strip job; every frame of the strip comes back and is written frame-numbered.
"""

import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import tifffile

from negpy.infrastructure.scanners.base import (
    ScannerBackend,
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    TransientScanError,
    ScannerUnavailable,
)
from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:5135"
_DPI = 2117  # 2000 px across the 24 mm frame height
_POLL_INTERVAL_S = 1.0
# The strip traversal streams ~400 MB for a 6-frame strip; used only to shape
# the progress fraction during the "scanning" phase.
_EXPECTED_STRIP_BYTES = 4.0e8


def _default_base_url() -> str:
    return os.environ.get("PSIX_URL", _DEFAULT_BASE_URL).rstrip("/")


class PsixBackend(ScannerBackend):
    """HTTP client for a running psix instance. Constructing is cheap and never
    touches the network — list_devices() returns [] while psix is down."""

    def __init__(self, base_url: str | None = None) -> None:
        self._base = (base_url or _default_base_url()).rstrip("/")
        self._strip_lock = threading.Lock()

    # ---- HTTP helpers ------------------------------------------------------

    def _get(self, path: str, timeout: float = 10.0):
        with urllib.request.urlopen(self._base + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, payload: dict, timeout: float = 15.0):
        req = urllib.request.Request(
            self._base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _download(self, path: str) -> np.ndarray:
        with urllib.request.urlopen(self._base + path, timeout=120.0) as resp:
            data = resp.read()
        return tifffile.imread(io.BytesIO(data))

    # ---- ScannerBackend ----------------------------------------------------

    def list_devices(self) -> list[ScannerDevice]:
        try:
            info = self._get("/api/device", timeout=3.0)
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("psix not reachable at %s: %s", self._base, exc)
            return []
        model = info.get("model")
        if not info.get("connected") or not model:
            return []
        caps = ScannerCapabilities(
            ir_channel=True,
            supported_dpi=(_DPI,),
            supported_depths=(16,),
            sources=(ScanMode.NEGATIVE,),
            max_area_mm=(36.0, 24.0),
            autofocus=False,
            can_eject=True,
            whole_strip=True,
        )
        return [
            ScannerDevice(
                id="psix",
                vendor="Kodak",
                model=f"Pakon {model} (psix)",
                capabilities=caps,
            )
        ]

    def refresh_devices(self) -> list[ScannerDevice]:
        return self.list_devices()

    def scan_strip(
        self,
        device_id: str,
        params: ScanParams,
        progress,
        cancel: threading.Event,
        cache: dict | None = None,
    ) -> list[ScanResult]:
        """One whole-strip acquisition: psix scans the loaded strip start to end,
        develops it, and the enumerated raw frames come back as ScanResults.

        `cache` (a dict) makes repeat calls with the same (device, IR) key return
        the cached strip — used by sessions so a per-frame batch hits the film
        once. Without it, every call is a fresh physical pass.
        """
        ir = bool(params.capture_ir)
        cache_key = (device_id, ir)
        if cache is not None:
            with self._strip_lock:
                cached = cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            roll = self._post("/api/rolls", {"name": "negpy " + datetime.now().strftime("%Y%m%d %H:%M")})["roll"]
            self._post(f"/api/rolls/{roll['id']}/scan", {"ir": ir})
        except (urllib.error.URLError, OSError) as exc:
            raise TransientScanError(f"psix at {self._base} did not accept the scan: {exc}") from exc

        # Poll the job to completion. psix develops the strip (frame slicing +
        # ICE) as part of the same job, so "busy" covers both legs.
        while True:
            if cancel.is_set():
                raise RuntimeError("scan cancelled")
            try:
                state = self._get("/api/runner", timeout=5.0)
            except (urllib.error.URLError, OSError) as exc:
                raise TransientScanError(f"lost contact with psix while scanning: {exc}") from exc
            phase = state.get("phase", "")
            if not state.get("busy"):
                if state.get("error") or phase == "failed":
                    raise RuntimeError(f"psix scan failed: {state.get('error') or phase}")
                progress(1.0, "Scanning")
                break
            if phase == "processing":
                progress(0.88, "Developing")
            elif phase == "scanning":
                fraction = min(1.0, float(state.get("bytes", 0)) / _EXPECTED_STRIP_BYTES)
                progress(0.02 + 0.83 * fraction, "Scanning")
            else:
                progress(0.01, phase.capitalize() or "Scanning")
            cancel.wait(_POLL_INTERVAL_S)

        try:
            listing = self._get(f"/api/rolls/{roll['id']}/raw", timeout=15.0)
        except (urllib.error.URLError, OSError) as exc:
            raise TransientScanError(f"lost contact with psix after the scan: {exc}") from exc
        entries = listing.get("frames", [])
        if not entries:
            raise RuntimeError("psix developed no frames from the strip — was film loaded?")

        device_model = "Pakon (psix)"
        try:
            device_model = self.list_devices()[0].model
        except Exception:  # noqa: BLE001 — label only
            pass

        results: list[ScanResult] = []
        for i, entry in enumerate(entries):
            if cancel.is_set():
                raise RuntimeError("scan cancelled")
            progress(0.9 + 0.1 * i / max(1, len(entries)), "Transferring")
            rgb = self._download(f"/api/rolls/{roll['id']}/raw/{entry['rgb']}")
            # ir=None: ICE is already applied to the RGB (the whole point of a
            # 4-channel psix scan), and psix's archived IR plane is unrescaled
            # against the isotropic RGB — handing it over would misalign NegPy's
            # dust tools. Re-add only alongside an aligned IR export in psix.
            results.append(ScanResult(rgb=rgb, ir=None, dpi=_DPI, device_model=device_model))

        if cache is not None:
            with self._strip_lock:
                cache[cache_key] = results
        return results

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress,
        cancel: threading.Event,
    ) -> ScanResult:
        """Protocol entry point: frame N (1-based) of a fresh strip pass."""
        results = self.scan_strip(device_id, params, progress, cancel)
        frame = params.frame or 1
        if frame < 1 or frame > len(results):
            raise RuntimeError(f"psix strip has {len(results)} frames; frame {frame} is out of range")
        return results[frame - 1]

    def open_session(self, device_id: str) -> ScannerSession:
        return _PsixSession(self, device_id)

    def eject(self, device_id: str) -> bool:
        try:
            self._post("/api/scanner/film", {"action": "eject"}, timeout=60.0)
            return True
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"psix eject failed: {exc}") from exc


class _PsixSession(ScannerSession):
    """Exclusive hold on the psix scanner for a batch: the first scan() runs one
    strip pass inside psix; further scan() calls serve the remaining frames from
    that pass without touching the film again."""

    def __init__(self, backend: PsixBackend, device_id: str) -> None:
        self.device_id = device_id
        self._backend = backend
        self._cache: dict[tuple, list[ScanResult]] = {}
        self._closed = False

    def scan(self, params: ScanParams, progress, cancel: threading.Event) -> ScanResult:
        if self._closed:
            raise ScannerUnavailable("psix session already closed")
        results = self._backend.scan_strip(
            self.device_id, params, progress, cancel, cache=self._cache)
        frame = params.frame or 1
        if frame < 1 or frame > len(results):
            raise RuntimeError(f"psix strip has {len(results)} frames; frame {frame} is out of range")
        return results[frame - 1]

    def eject(self) -> bool:
        if self._closed:
            return False
        return self._backend.eject(self.device_id)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "_PsixSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
