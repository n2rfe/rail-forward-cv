"""Ranging checks on synthetic rails projected through a known pinhole camera."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from railcv.ranging import MPH, CameraModel, estimate_range, stopping_distance

FX = FY = 1786.0
CX, CY = 960.0, 540.0
G, H = 1.505, 1.2  # gauge (m), camera height (m)


def synth_rails(z_max=45.0, z_min=4.0, n=64, noise_px=0.0, seed=0):
    """Level straight track, camera centered and looking straight ahead."""
    rng = np.random.default_rng(seed)
    z = np.linspace(z_min, z_max, n)
    y = CY + FY * H / z
    xl = CX + FX * (-G / 2) / z + rng.normal(0, noise_px, n)
    xr = CX + FX * (+G / 2) / z + rng.normal(0, noise_px, n)
    return np.column_stack([xl, y]), np.column_stack([xr, y]), z


def cam():
    return CameraModel(FX, FY, CX, CY, gauge_m=G)


def test_exact_geometry_recovers_range_height_horizon():
    left, right, z = synth_rails()
    r = estimate_range(left, right, cam())
    assert r.clear_m == pytest.approx(z.max(), rel=1e-6)
    assert r.cam_height_m == pytest.approx(H, rel=1e-6)
    assert r.horizon_y == pytest.approx(CY, abs=1e-3)
    assert np.allclose(r.z_raw, z)


def test_fit_beats_raw_under_pixel_noise():
    errs_fit, errs_raw = [], []
    for s in range(50):
        left, right, z = synth_rails(noise_px=0.7, seed=s)
        r = estimate_range(left, right, cam())
        far = np.argmax(z)
        errs_fit.append(abs(r.clear_m - z[far]))
        errs_raw.append(abs(r.z_raw[far] - z[far]))
    assert np.mean(errs_fit) < 0.5 * np.mean(errs_raw)


def test_one_pixel_error_scale():
    r = estimate_range(*synth_rails(z_max=45)[:2], cam())
    assert r.clear_err_m == pytest.approx(45**2 / (FX * G), rel=1e-6)  # ~0.75 m


def test_clipped_anchors_are_rejected():
    left, right, _ = synth_rails()
    crop = (int(left[:, 0].min()) + 30, 0, 1919, 1079)  # crop cuts the near left rail
    left[:, 0] = np.maximum(left[:, 0], crop[0])  # TEP-Net clips x to the crop
    r = estimate_range(left, right, cam(), crop=crop)
    assert not r.valid[0] and r.valid[-1]
    assert r.cam_height_m == pytest.approx(H, rel=1e-3)


def test_row_at_z_roundtrip():
    r = estimate_range(*synth_rails()[:2], cam())
    for zq in (10, 20, 30):
        assert r.z_at_row(r.row_at_z(zq)) == pytest.approx(zq)


def test_degenerate_inputs():
    assert estimate_range(np.empty((0, 2)), np.empty((0, 2)), cam()).clear_m is None


def test_stopping_distance_20mph():
    v = 20 * MPH
    assert stopping_distance(v, 2.0, 0.5) == pytest.approx(v * 0.5 + v**2 / 4.0)
    assert 24 < stopping_distance(v, 2.0, 0.5) < 25
