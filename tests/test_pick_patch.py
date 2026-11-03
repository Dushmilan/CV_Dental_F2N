import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

import pick_patch


def test_snap_patch_snaps_size_and_keeps_center():
    assert pick_patch.snap_patch(10, 20, 74, 90, 265, 370) == (10, 19, 64, 72)


def test_snap_patch_grows_a_tiny_drag_to_the_minimum():
    x, y, w, h = pick_patch.snap_patch(100, 100, 105, 103, 265, 370)
    assert (w, h) == (32, 32)
    assert abs((x + w / 2) - 102.5) <= 1
    assert abs((y + h / 2) - 101.5) <= 1


def test_snap_patch_clamps_a_drag_outside_the_image():
    x, y, w, h = pick_patch.snap_patch(-50, -50, 500, 400, 265, 370)
    assert w % 8 == 0 and h % 8 == 0
    assert 0 <= x and x + w <= 370
    assert 0 <= y and y + h <= 265


def test_snap_patch_rejects_an_image_smaller_than_the_grid():
    with pytest.raises(ValueError):
        pick_patch.snap_patch(0, 0, 4, 4, 4, 4)


def test_hu_conversion_round_trips_inside_the_window():
    hu = np.array([[-2000, -1000, 0, 2000, 3000]], dtype=np.float32)
    unit = pick_patch.hu_to_unit(hu, (-1000, 2000))
    back = pick_patch.unit_to_hu(unit, (-1000, 2000))
    assert np.allclose(back, np.clip(hu, -1000, 2000), atol=1e-4)


import torch


def test_pad_and_crop_round_trip():
    arr = np.arange(70 * 70, dtype=np.float32).reshape(70, 70)
    padded, (top, left) = pick_patch.pad_to_multiple(arr, 16)
    assert padded.shape == (80, 80)
    assert np.array_equal(padded[top:top + 70, left:left + 70], arr)


def test_train_patch_is_deterministic_and_finite():
    patch = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    kwargs = dict(device=torch.device("cpu"), epochs=2, lr=1e-3, lambda_weight=350, seed=77)
    first, first_losses, first_info = pick_patch.train_patch(patch, **kwargs)
    second, second_losses, second_info = pick_patch.train_patch(patch, **kwargs)
    assert first.shape == (32, 32)
    assert np.isfinite(first).all()
    assert len(first_losses) == 2
    assert np.array_equal(first, second)
    assert first_info["params"] == second_info["params"]


import json


def test_compute_metrics_on_a_noop():
    patch = np.full((8, 8), 100.0, dtype=np.float32)
    metrics = pick_patch.compute_metrics(patch, patch, (-1000, 2000))
    assert metrics["mean_abs_change_hu"] == 0.0
    assert metrics["max_abs_change_hu"] == 0.0
    assert np.isinf(metrics["psnr_input"])


def test_save_run_writes_arrays_config_and_curve(tmp_path):
    patch = np.zeros((32, 32), dtype=np.float32)
    out = pick_patch.save_run(
        tmp_path / "run",
        patch_input_hu=patch,
        patch_denoised_hu=patch,
        losses=[1.0, 0.5],
        config={"slice": 106, "coords": [1, 2, 32, 32]},
    )
    for name in ("patch_input_hu.npy", "patch_denoised_hu.npy", "loss_curve.png", "run_config.json"):
        assert (out / name).exists()
    assert json.loads((out / "run_config.json").read_text())["slice"] == 106


def test_extents_to_patch_reorders_and_handles_cancel():
    assert pick_patch.extents_to_patch((10, 74, 20, 90), 265, 370) == (10, 19, 64, 72)
    assert pick_patch.extents_to_patch(None, 265, 370) is None


def test_select_patch_requires_a_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError):
        pick_patch.select_patch(np.zeros((8, 8), dtype=np.float32), (-200, 400))


def test_select_patch_accepts_a_dragged_rectangle(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    slice_hu = np.zeros((265, 370), dtype=np.float32)

    def fake_show():
        fig = pick_patch.plt.gcf()
        ax = fig.axes[0]
        (x0, y0), (x1, y1) = ax.transData.transform([(40, 40), (120, 120)])
        MouseEvent("button_press_event", fig.canvas, x0, y0, button=1)._process()
        MouseEvent("motion_notify_event", fig.canvas, x1, y1, button=1)._process()
        MouseEvent("button_release_event", fig.canvas, x1, y1, button=1)._process()
        KeyEvent("key_press_event", fig.canvas, "enter")._process()

    monkeypatch.setattr(pick_patch.plt, "show", fake_show)

    assert pick_patch.select_patch(slice_hu, (-200, 400)) == (40, 40, 80, 80)


def test_validate_explicit_patch_rejects_bad_input():
    assert pick_patch.validate_explicit_patch(10, 20, 64, 64, 265, 370) == (10, 20, 64, 64)
    with pytest.raises(ValueError):
        pick_patch.validate_explicit_patch(10, 20, 63, 64, 265, 370)
    with pytest.raises(ValueError):
        pick_patch.validate_explicit_patch(340, 240, 64, 64, 265, 370)


def test_validate_slice_index_rejects_out_of_range():
    pick_patch.validate_slice_index(168, 169)
    with pytest.raises(ValueError):
        pick_patch.validate_slice_index(169, 169)


def test_parse_args_accepts_a_patch_bypass():
    args = pick_patch.parse_args(["--slice", "106", "--patch", "120", "140", "64", "64"])
    assert args.slice == 106
    assert args.patch == [120, 140, 64, 64]
    assert args.lambda_weight == 180.0


def test_snap_patch_rejects_an_image_below_the_minimum_patch():
    for size in (16, 24):
        with pytest.raises(ValueError):
            pick_patch.snap_patch(0, 0, size, size, size, size)


def test_validate_explicit_patch_enforces_the_minimum_size():
    with pytest.raises(ValueError):
        pick_patch.validate_explicit_patch(0, 0, 8, 8, 265, 370)
    with pytest.raises(ValueError):
        pick_patch.validate_explicit_patch(0, 0, 24, 64, 265, 370)
    assert pick_patch.validate_explicit_patch(0, 0, 32, 32, 265, 370) == (0, 0, 32, 32)


def test_parse_args_rejects_an_invalid_minimum_patch():
    with pytest.raises(SystemExit):
        pick_patch.parse_args(["--min-patch", "50"])
    with pytest.raises(SystemExit):
        pick_patch.parse_args(["--min-patch", "24"])


def _fake_train(patch, **kwargs):
    sigmas = [np.zeros((*patch.shape, 3), dtype=np.float32)]
    return patch.copy(), [1.0], {
        "loss": 1.0,
        "train_seconds": 0.01,
        "params": 10,
        "sigmas": sigmas,
    }


def test_main_aborts_on_non_finite_output_before_writing(tmp_path, monkeypatch):
    np.save(tmp_path / "volume.npy", np.zeros((2, 40, 40), dtype=np.float32))

    def fake_train(patch, **kwargs):
        out = np.full(patch.shape, np.nan, dtype=np.float32)
        return out, [1.0], {
            "loss": 1.0,
            "train_seconds": 0.01,
            "params": 10,
            "sigmas": [np.zeros((*patch.shape, 3), dtype=np.float32)],
        }

    monkeypatch.setattr(pick_patch, "train_patch", fake_train)
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError):
        pick_patch.main([
            "--input", str(tmp_path / "volume.npy"),
            "--slice", "0",
            "--patch", "0", "0", "32", "32",
            "--epochs", "1",
            "--out", str(out_dir),
        ])
    assert not out_dir.exists()


def test_make_result_figure_shows_only_when_requested(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(pick_patch.plt, "show", lambda: shown.append(True))
    patch = np.zeros((32, 32), dtype=np.float32)
    kwargs = dict(
        slice_hu=np.zeros((40, 40), dtype=np.float32),
        patch_input_hu=patch,
        patch_denoised_hu=patch,
        sigmas=[np.zeros((32, 32, 3), dtype=np.float32)],
        coords=(0, 0, 32, 32),
        metrics={
            "mean_abs_change_hu": 0.0,
            "max_abs_change_hu": 0.0,
            "psnr_input": 50.0,
            "ssim_input": 1.0,
        },
        display_window=(-200.0, 400.0),
    )
    pick_patch.make_result_figure(**kwargs, save_path=tmp_path / "a.png")
    assert shown == []
    pick_patch.make_result_figure(**kwargs, save_path=tmp_path / "b.png", show=True)
    assert shown == [True]


def test_main_shows_result_only_after_gui_selection(tmp_path, monkeypatch):
    np.save(tmp_path / "volume.npy", np.zeros((2, 40, 40), dtype=np.float32))
    monkeypatch.setattr(pick_patch, "train_patch", _fake_train)
    monkeypatch.setattr(pick_patch, "select_patch", lambda *a, **k: (0, 0, 32, 32))
    captured = {}

    def fake_figure(**kwargs):
        captured.update(kwargs)
        return kwargs["save_path"]

    monkeypatch.setattr(pick_patch, "make_result_figure", fake_figure)

    common = [
        "--input", str(tmp_path / "volume.npy"),
        "--slice", "0",
        "--epochs", "1",
        "--patch", "0", "0", "32", "32",
    ]
    pick_patch.main([*common, "--out", str(tmp_path / "explicit")])
    assert captured["show"] is False

    gui = ["--input", str(tmp_path / "volume.npy"), "--slice", "0", "--epochs", "1"]
    pick_patch.main([*gui, "--out", str(tmp_path / "gui")])
    assert captured["show"] is True
