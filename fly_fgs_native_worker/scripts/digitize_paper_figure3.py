#!/usr/bin/env python3
"""Digitize Reichardt et al. 1983 Figure 3 from the supplied PDF.

The script renders PDF page 5 at 600 dpi, fits each axis from every visible
tick, traces each curve independently in both x directions, and records a
per-sample uncertainty.  It never smooths, time-warps, shifts, or rescales a
trace to a simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image


SCHEMA_VERSION = "reichardt_1983_figure3_digitization.v1"
PDF_SHA256 = "9159ec24548e1ee0eaf4edca0d4790d7656890d2fb611dd0b7d27cc85a6f05d7"
PANELS: Mapping[str, Mapping[str, Any]] = {
    "R83_Fig3a_0_to_90": {
        "panel": "3a",
        "bounds_px": (687, 1777, 661, 1503),
        "time_ticks": tuple(zip((687, 909, 1128, 1342, 1564, 1777), (0, .4, .8, 1.2, 1.6, 2.0))),
        "torque_ticks": tuple(zip((738, 880, 1023, 1161, 1300, 1439), (.8, .6, .4, .2, 0, -.2))),
    },
    "R83_Fig3b_0_to_270": {
        "panel": "3b",
        "bounds_px": (2048, 3124, 642, 1510),
        "time_ticks": tuple(zip((2048, 2262, 2478, 2697, 2908, 3124), (0, .4, .8, 1.2, 1.6, 2.0))),
        "torque_ticks": tuple(zip((750, 895, 1026, 1169, 1303, 1442), (.8, .6, .4, .2, 0, -.2))),
    },
    "R83_Fig3c_0_to_180": {
        "panel": "3c",
        "bounds_px": (3414, 4485, 810, 1513),
        "time_ticks": tuple(zip((3414, 3523, 3631, 3739, 3844, 3953, 4061, 4165, 4275, 4383, 4485), (0, .4, .8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0))),
        "torque_ticks": tuple(zip((894, 1035, 1173, 1315, 1458), (.2, .1, 0, -.1, -.2))),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dark_runs(column: np.ndarray, threshold: int) -> List[Tuple[int, int, float, int]]:
    indices = np.flatnonzero(column < threshold)
    result = []
    for chunk in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if len(chunk):
            result.append(
                (int(chunk[0]), int(chunk[-1]), float(np.mean(chunk)), len(chunk))
            )
    return result


def _trace_one_direction(
    image: np.ndarray,
    bounds: Sequence[int],
    *,
    threshold: int,
    direction: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = (int(value) for value in bounds)
    xs = np.arange(x0 + 12, x1 - 12, dtype=int)
    sequence = xs if direction > 0 else xs[::-1]
    centerlines = []
    thicknesses = []
    previous = None
    for x in sequence:
        runs = _dark_runs(image[y0 + 20 : y1 - 20, x], threshold)
        if not runs:
            raise RuntimeError("curve tracing found an empty rendered column")
        if previous is None:
            selected = max(
                runs,
                key=lambda run: (
                    run[3],
                    -abs(run[2] - 0.5 * (y1 - y0)),
                ),
            )
        else:
            selected = min(
                runs,
                key=lambda run: abs(run[2] - previous) + max(0, 3 - run[3]) * 2,
            )
        previous = selected[2]
        centerlines.append(selected[2] + y0 + 20)
        thicknesses.append(selected[3])
    if direction < 0:
        centerlines.reverse()
        thicknesses.reverse()
    return xs, np.asarray(centerlines), np.asarray(thicknesses)


def _affine_receipt(points: Sequence[Tuple[int, float]]) -> Mapping[str, Any]:
    source_pixels = np.asarray([point[0] for point in points], dtype=float)
    values = np.asarray([point[1] for point in points], dtype=float)
    # The printed scan has small local geometric distortions. Use every tick to
    # map its source coordinate onto an evenly spaced rectified pixel axis,
    # then apply the declared affine physical calibration in that rectified
    # frame. Source-affine residual remains explicit digitization uncertainty.
    rectified_pixels = np.linspace(
        source_pixels[0], source_pixels[-1], len(source_pixels)
    )
    slope, intercept = np.polyfit(rectified_pixels, values, 1)
    reconstructed_pixels = (values - intercept) / slope
    residual = reconstructed_pixels - rectified_pixels
    source_slope, source_intercept = np.polyfit(source_pixels, values, 1)
    source_reconstructed = (values - source_intercept) / source_slope
    source_residual = source_reconstructed - source_pixels
    return {
        "control_points": [
            {
                "source_pixel": int(source_pixel),
                "rectified_pixel": float(rectified_pixel),
                "value": float(value),
            }
            for source_pixel, rectified_pixel, value in zip(
                source_pixels, rectified_pixels, values
            )
        ],
        "source_to_rectified_method": "piecewise linear through every visible tick",
        "value_per_pixel": float(slope),
        "value_at_pixel_zero": float(intercept),
        "landmark_rms_error_px": float(np.sqrt(np.mean(residual ** 2))),
        "landmark_max_error_px": float(np.max(np.abs(residual))),
        "landmark_roundtrip_within_0_25px": bool(
            np.max(np.abs(residual)) <= 0.25
        ),
        "source_affine_rms_error_px": float(
            np.sqrt(np.mean(source_residual ** 2))
        ),
        "source_affine_max_error_px": float(np.max(np.abs(source_residual))),
    }


def _rectify_pixels(pixels: np.ndarray, receipt: Mapping[str, Any]) -> np.ndarray:
    points = receipt["control_points"]
    source = np.asarray([point["source_pixel"] for point in points], dtype=float)
    rectified = np.asarray(
        [point["rectified_pixel"] for point in points], dtype=float
    )
    return np.interp(np.asarray(pixels, dtype=float), source, rectified)


def digitize(pdf_path: Path, output_path: Path) -> Mapping[str, Any]:
    pdf_path = pdf_path.resolve()
    if _sha256(pdf_path) != PDF_SHA256:
        raise ValueError("paper PDF SHA-256 does not match the reviewed source")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="figure3-600dpi-") as directory:
        rendered = Path(directory) / "page5.png"
        subprocess.run(
            (
                "gs",
                "-q",
                "-dNOPAUSE",
                "-dBATCH",
                "-dSAFER",
                "-sDEVICE=pnggray",
                "-r600",
                "-dFirstPage=5",
                "-dLastPage=5",
                "-sOutputFile={}".format(rendered),
                str(pdf_path),
            ),
            check=True,
        )
        page = Image.open(rendered).convert("L")
        image = np.asarray(page)
        if page.size != (4983, 6542):
            raise ValueError("600 dpi page raster dimensions changed")
        panels: Dict[str, Any] = {}
        for protocol_id, definition in PANELS.items():
            bounds = definition["bounds_px"]
            time_fit = _affine_receipt(definition["time_ticks"])
            torque_fit = _affine_receipt(definition["torque_ticks"])
            x_forward, y_forward, thickness_forward = _trace_one_direction(
                image, bounds, threshold=150, direction=1
            )
            x_reverse, y_reverse, thickness_reverse = _trace_one_direction(
                image, bounds, threshold=190, direction=-1
            )
            if not np.array_equal(x_forward, x_reverse):
                raise RuntimeError("independent trace grids disagree")
            x = _rectify_pixels(x_forward.astype(float), time_fit)
            y_forward_rectified = _rectify_pixels(y_forward, torque_fit)
            y_reverse_rectified = _rectify_pixels(y_reverse, torque_fit)
            time_s = time_fit["value_per_pixel"] * x + time_fit["value_at_pixel_zero"]
            torque_forward = (
                torque_fit["value_per_pixel"] * y_forward_rectified
                + torque_fit["value_at_pixel_zero"]
            )
            torque_reverse = (
                torque_fit["value_per_pixel"] * y_reverse_rectified
                + torque_fit["value_at_pixel_zero"]
            )
            torque_center = 0.5 * (torque_forward + torque_reverse)
            half_line = (
                0.25
                * (thickness_forward + thickness_reverse)
                * abs(torque_fit["value_per_pixel"])
            )
            axis_rms = (
                torque_fit["source_affine_rms_error_px"]
                * abs(torque_fit["value_per_pixel"])
            )
            half_disagreement = 0.5 * np.abs(torque_forward - torque_reverse)
            uncertainty = np.maximum.reduce(
                (
                    half_line,
                    np.full_like(half_line, axis_rms),
                    half_disagreement,
                )
            )
            crop_name = "figure3_panel_{}_600dpi.png".format(
                str(definition["panel"])[-1]
            )
            crop_path = output_path.parent / crop_name
            x0, x1, y0, y1 = bounds
            page.crop((x0, y0, x1, y1)).save(crop_path)
            panels[protocol_id] = {
                "panel": definition["panel"],
                "crop": {
                    "file": crop_name,
                    "sha256": _sha256(crop_path),
                    "bounds_page_px": list(bounds),
                },
                "time_axis": time_fit,
                "torque_axis_dyne_cm": torque_fit,
                "trace_methods": (
                    "forward column-run centerline at grayscale threshold 150",
                    "reverse column-run centerline at grayscale threshold 190",
                ),
                "time_s": time_s.tolist(),
                "torque_dyne_cm": torque_center.tolist(),
                "digitization_uncertainty_dyne_cm": uncertainty.tolist(),
                "independent_trace_1_dyne_cm": torque_forward.tolist(),
                "independent_trace_2_dyne_cm": torque_reverse.tolist(),
            }
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "citation": "Reichardt et al. (1983), Biological Cybernetics 46:1-12",
            "doi": "10.1007/BF00595226",
            "pdf_filename": pdf_path.name,
            "pdf_sha256": PDF_SHA256,
            "pdf_page": 5,
            "render_dpi": 600,
            "rendered_page_size_px": [4983, 6542],
        },
        "paper_curve_semantics": {
            "sweeps_per_curve": 100,
            "individual": "one typical fly",
            "paper_reported_sem": False,
            "acquisition_rate_reported": False,
            "filter_reported": False,
        },
        "uncertainty_definition": (
            "maximum of transformed half-line thickness, y-axis affine RMS, "
            "and half the independent-trace disagreement"
        ),
        "no_simulation_fitting": True,
        "panels": panels,
    }
    output_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = digitize(args.pdf, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "panels": list(payload["panels"]),
                "pdf_sha256": payload["source"]["pdf_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
