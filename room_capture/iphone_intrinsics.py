"""iPhone 16 Pro intrinsics builder (main 1x + ultra-wide 0.5x).

Priority order for recovering (fx, fy, cx, cy):
    1. EXIF `FocalLengthIn35mmFormat` -> horizontal FoV -> pixel focal.
    2. EXIF `FocalLength` + `FocalPlaneXResolution` -> pixel focal directly.
    3. Hard-coded iPhone 16 Pro per-lens fallback (main 24mm-equiv or
       ultra-wide 13mm-equiv).

The returned K auto-scales for any (W, H) - works whether DA3 resizes to 504px
or the user feeds full 48 MP frames.

Note on the ultra-wide lens: it has noticeable barrel distortion. Apple's ISP
corrects most of it when writing JPEG/HEIC, but DA3 still assumes a pure
pinhole model, so pose/depth is a bit noisier than with the main lens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image, ExifTags


# 35mm-equivalent focal length per iPhone 16 Pro lens.
# HFoV = 2 * atan(36 / (2 * f35))  using a 36 mm-wide full-frame sensor.
_LENS_F35 = {
    "main": 24.0,       # 1x
    "ultrawide": 13.0,  # 0.5x
}

LensName = Literal["main", "ultrawide", "auto"]


def _hfov_from_f35(f35_mm: float) -> float:
    return 2.0 * math.degrees(math.atan(36.0 / (2.0 * f35_mm)))


def _focal_divisor_from_hfov(hfov_deg: float) -> float:
    """Reverse identity: fx = W / (2 * tan(HFoV/2))."""
    return 2.0 * math.tan(math.radians(hfov_deg) / 2.0)


_IPHONE16_PRO_MAIN_HFOV_DEG = _hfov_from_f35(_LENS_F35["main"])           # ~73.74 deg
_IPHONE16_PRO_UW_HFOV_DEG = _hfov_from_f35(_LENS_F35["ultrawide"])        # ~108.41 deg

# Threshold in 35mm-equiv focal length for auto-classifying main vs ultra-wide.
# Anything <= 18mm-equiv is treated as the 0.5x ultra-wide lens.
_ULTRAWIDE_F35_THRESHOLD_MM = 18.0


@dataclass
class IntrinsicsResult:
    K: np.ndarray             # (3, 3) float32
    source: str               # e.g. "exif_35mm(ultrawide)" | "fallback_iphone16pro(main)"
    hfov_deg: float
    lens: str                 # "main" | "ultrawide"


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

_EXIF_TAG_NAME_TO_ID = {v: k for k, v in ExifTags.TAGS.items()}


def _exif_get(exif: dict[int, Any] | None, name: str) -> Any:
    if exif is None:
        return None
    tag_id = _EXIF_TAG_NAME_TO_ID.get(name)
    if tag_id is None:
        return None
    return exif.get(tag_id)


def _to_float(v: Any) -> float | None:
    """Coerce EXIF numeric types (IFDRational, tuple, Fraction, int, str) to float."""
    if v is None:
        return None
    try:
        # PIL's IFDRational and fractions.Fraction both support float().
        return float(v)
    except (TypeError, ValueError):
        pass
    if isinstance(v, tuple) and len(v) == 2:
        num, den = v
        try:
            return float(num) / float(den) if float(den) != 0 else None
        except (TypeError, ValueError):
            return None
    return None


def _read_exif(image: Image.Image) -> dict[int, Any] | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    merged: dict[int, Any] = dict(exif)
    # Merge the ExifIFD sub-block (where FocalLengthIn35mmFormat usually lives).
    exif_ifd_id = _EXIF_TAG_NAME_TO_ID.get("ExifOffset")
    if exif_ifd_id is not None:
        try:
            sub = exif.get_ifd(exif_ifd_id)
        except Exception:
            sub = None
        if sub:
            merged.update(sub)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _classify_lens_from_f35(f35_mm: float) -> str:
    """Map a 35mm-equivalent focal length to a lens label."""
    return "ultrawide" if f35_mm <= _ULTRAWIDE_F35_THRESHOLD_MM else "main"


def iphone16_pro_fallback_K(
    width: int,
    height: int,
    lens: Literal["main", "ultrawide"] = "main",
) -> np.ndarray:
    """Hard-coded iPhone 16 Pro K for the chosen lens (no EXIF required)."""
    if lens not in _LENS_F35:
        raise ValueError(f"unknown lens {lens!r}; expected 'main' or 'ultrawide'")
    hfov_deg = _hfov_from_f35(_LENS_F35[lens])
    divisor = _focal_divisor_from_hfov(hfov_deg)
    fx = fy = float(width) / divisor
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def build_K(
    image: Image.Image,
    lens: LensName = "auto",
) -> IntrinsicsResult:
    """Construct a (3, 3) camera intrinsic matrix for a single iPhone photo.

    Args:
        image: PIL image (EXIF preserved).
        lens:  "main" (1x / 24mm-equiv), "ultrawide" (0.5x / 13mm-equiv), or
               "auto". In "auto" mode EXIF is trusted when present, and the
               fallback assumes the main lens. When lens is explicitly set to
               "main" or "ultrawide", it is used both for classifying the EXIF
               result and as the fallback profile.
    """
    if lens not in ("main", "ultrawide", "auto"):
        raise ValueError(f"unknown lens {lens!r}")

    W, H = image.size
    exif = _read_exif(image)

    # --- Strategy 1: EXIF FocalLengthIn35mmFormat ---------------------------
    f35 = _to_float(_exif_get(exif, "FocalLengthIn35mmFilm"))
    if f35 is None:
        f35 = _to_float(_exif_get(exif, "FocalLengthIn35mmFormat"))
    if f35 and f35 > 0:
        hfov_rad = 2.0 * math.atan(36.0 / (2.0 * f35))
        fx = float(W) / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = (float(W) - 1.0) / 2.0
        cy = (float(H) - 1.0) / 2.0
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        detected_lens = _classify_lens_from_f35(f35)
        # If the caller explicitly pinned a lens, respect their label even if
        # EXIF disagrees (handy for third-party photos with stripped tags).
        lens_label = lens if lens != "auto" else detected_lens
        return IntrinsicsResult(
            K=K,
            source=f"exif_35mm({detected_lens})",
            hfov_deg=math.degrees(hfov_rad),
            lens=lens_label,
        )

    # --- Strategy 2: EXIF FocalLength + FocalPlaneXResolution --------------
    focal_mm = _to_float(_exif_get(exif, "FocalLength"))
    fpxr = _to_float(_exif_get(exif, "FocalPlaneXResolution"))
    fpyr = _to_float(_exif_get(exif, "FocalPlaneYResolution"))
    fpru = _exif_get(exif, "FocalPlaneResolutionUnit")  # 2=inch, 3=cm, 4=mm
    if focal_mm and focal_mm > 0 and fpxr and fpxr > 0:
        # FocalPlaneXResolution is pixels-per-unit; convert to pixels-per-mm.
        unit_to_mm = {2: 25.4, 3: 10.0, 4: 1.0}.get(int(fpru) if fpru else 2, 25.4)
        px_per_mm_x = fpxr / unit_to_mm
        px_per_mm_y = (fpyr / unit_to_mm) if fpyr and fpyr > 0 else px_per_mm_x
        fx = focal_mm * px_per_mm_x
        fy = focal_mm * px_per_mm_y
        cx = (float(W) - 1.0) / 2.0
        cy = (float(H) - 1.0) / 2.0
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        hfov_deg = math.degrees(2.0 * math.atan(float(W) / (2.0 * fx)))
        # Classify by inferred HFoV (wider than ~90 deg -> ultra-wide).
        detected_lens = "ultrawide" if hfov_deg >= 90.0 else "main"
        lens_label = lens if lens != "auto" else detected_lens
        return IntrinsicsResult(
            K=K,
            source=f"exif_focal_plane({detected_lens})",
            hfov_deg=hfov_deg,
            lens=lens_label,
        )

    # --- Strategy 3: iPhone 16 Pro fallback (per lens) ----------------------
    fallback_lens: Literal["main", "ultrawide"] = (
        "main" if lens == "auto" else lens  # type: ignore[assignment]
    )
    K = iphone16_pro_fallback_K(W, H, lens=fallback_lens)
    return IntrinsicsResult(
        K=K,
        source=f"fallback_iphone16pro({fallback_lens})",
        hfov_deg=_hfov_from_f35(_LENS_F35[fallback_lens]),
        lens=fallback_lens,
    )


def scale_K(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Rescale a K matrix when an image is resized from src to dst resolution."""
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    K_out = K.copy().astype(np.float32)
    K_out[0, 0] *= sx
    K_out[0, 2] *= sx
    K_out[1, 1] *= sy
    K_out[1, 2] *= sy
    return K_out
