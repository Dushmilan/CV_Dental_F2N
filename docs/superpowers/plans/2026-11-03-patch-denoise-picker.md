# Patch-level Filter2Noise Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `pick_patch.py`, a local matplotlib tool that lets the user drag a region on a CBCT slice and trains a fresh Filter2Noise model on that patch alone.

**Architecture:** One self-contained script imports `DenoisingPipeline` and `LossFunction` from `filter2noise.py`. Pure helpers handle patch geometry, HU conversion, padding, metrics, and saving; a thin GUI layer collects a rectangle; `main()` wires selection, training, metrics, and output. Tests cover every helper and a two-epoch training run.

**Tech Stack:** Python 3.12, NumPy 1.26.4, PyTorch 2.14.1+cpu, matplotlib 3.6.3 (TkAgg), scikit-image, pytest 8.4.2.

**Spec:** `docs/superpowers/specs/2026-11-03-patch-denoise-picker-design.md`

## Global Constraints

- Interpreter and deps live in `.venv_cpu`; run everything with `.venv_cpu/bin/python`.
- Do not modify `run_cbct.py` or `filter2noise.py`.
- Do not import `run_cbct.py`; it forces the matplotlib `Agg` backend at import time.
- Defaults copied from the spec: `--input /home/dushmilan/Desktop/data.npy`, `--slice 106`, `--epochs 500`, `--lr 1e-3`, `--lambda-weight 350`, `--num-stages 2`, `--patch-size 8`, `--window -1000 2000`, `--display-window -200 400`, `--seed 77`, `--device auto`, `--out runs/patch_test`, `--min-patch 32`.
- Patch coordinates are `(x, y, w, h)`, with `x` = column and `y` = row of the top-left corner.
- Source patch `w` and `h` must be multiples of 8 and at least 32; padding before training makes them multiples of 16.
- `run_config.json` must record `weights: "random_init"` and `trained_on: "selected_patch_only"`.

## Review Focus

1. A drag that starts or ends outside the slice: result stays inside `0..width` / `0..height`, never negative.
2. A very small drag, or a slice smaller than the 8-pixel grid: grown to 32×32, or a clear `ValueError` when the image cannot fit.
3. Entering no rectangle and pressing Esc: clean exit 0 and no output files.
4. `--patch` with out-of-range or non-multiple-of-8 coordinates: `ValueError` before any training.
5. A slice index outside the volume: clear `ValueError` before loading pixels.

---

### Task 1: Patch geometry and HU conversion

**Files:**
- Create: `pick_patch.py`
- Create: `tests/test_pick_patch.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `snap_patch(x0: float, y0: float, x1: float, y1: float, height: int, width: int, multiple: int = 8, min_size: int = 32) -> tuple[int, int, int, int]`
  - `hu_to_unit(image_hu: np.ndarray, window: tuple[float, float]) -> np.ndarray`
  - `unit_to_hu(image_unit: np.ndarray, window: tuple[float, float]) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pick_patch.py
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pick_patch'`.

- [ ] **Step 3: Implement the geometry and HU helpers**

`snap_patch` orders the coordinates, clamps them to the image, rounds the size to the nearest multiple of `multiple` with `min_size` as the floor, reduces the size to the largest multiple that fits, and re-centres the snapped rectangle on the original drag centre before clamping the origin. Raise `ValueError` when `height < multiple` or `width < multiple`.

`hu_to_unit` clips `(image - lo) / (hi - lo)` to `[0, 1]` and returns float32; `unit_to_hu` is the inverse. Raise `ValueError` when `hi <= lo`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pick_patch.py tests/test_pick_patch.py
git commit -m "feat: add patch geometry and HU helpers"
```

---

### Task 2: Padding and patch training

**Files:**
- Modify: `pick_patch.py`
- Modify: `tests/test_pick_patch.py`

**Interfaces:**
- Consumes: `hu_to_unit`, `unit_to_hu` from Task 1.
- Produces:
  - `pad_to_multiple(image: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]` returning `(padded, (top, left))`
  - `resolve_device(name: str) -> torch.device`
  - `train_patch(patch_unit: np.ndarray, *, device: torch.device, epochs: int, lr: float, lambda_weight: float, num_stages: int = 2, patch_size: int = 8, seed: int = 77, log_every: int = 50) -> tuple[np.ndarray, list[float], dict]`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_pick_patch.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: FAIL with `AttributeError: module 'pick_patch' has no attribute 'pad_to_multiple'`.

- [ ] **Step 3: Implement padding and training**

`pad_to_multiple` mirrors `run_cbct.py`: compute the target size with `ceil`, split the padding around the centre, and use `np.pad(..., mode="reflect")`.

`resolve_device("auto")` returns CUDA when `torch.cuda.is_available()`, otherwise CPU; any other name is passed to `torch.device`.

`train_patch` converts the 2D array to a `(1, 1, H, W)` tensor, seeds with `random.seed`, `np.random.seed`, `torch.manual_seed`, builds a fresh `DenoisingPipeline(num_stages=num_stages, patch_size=patch_size)`, materialises its `LazyLinear` parameters with one forward under `no_grad`, then trains with `AdamW(lr=lr, weight_decay=0.01)` and `OneCycleLR(max_lr=lr, epochs=epochs, steps_per_epoch=1)`. Append `float(loss.detach())` each epoch; raise `RuntimeError(f"Non-finite loss at epoch {epoch + 1}")` when the loss is not finite. After training, run one forward with `return_sigmas=True` and return the denoised unit image, the loss list, and an info dict with `loss`, `train_seconds`, `params`, and `sigmas` (a list of `(H, W, 3)` NumPy arrays).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add pick_patch.py tests/test_pick_patch.py
git commit -m "feat: add patch padding and training"
```

---

### Task 3: Metrics and run saving

**Files:**
- Modify: `pick_patch.py`
- Modify: `tests/test_pick_patch.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `compute_metrics(patch_hu: np.ndarray, denoised_hu: np.ndarray, window: tuple[float, float]) -> dict` with keys `mean_abs_change_hu`, `max_abs_change_hu`, `psnr_input`, `ssim_input`
  - `save_run(out_dir: Path, *, patch_input_hu: np.ndarray, patch_denoised_hu: np.ndarray, losses: list[float], config: dict) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_pick_patch.py
import json
from pathlib import Path


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: FAIL with `AttributeError: module 'pick_patch' has no attribute 'compute_metrics'`.

- [ ] **Step 3: Implement metrics and saving**

`compute_metrics` computes the absolute HU difference, its mean and maximum, then `skimage.metrics.peak_signal_noise_ratio` and `structural_similarity` against `patch_hu` using `data_range=hi - lo`. Cast all values to plain floats.

`save_run` creates the directory, writes `patch_input_hu.npy`, `patch_denoised_hu.npy`, `run_config.json` (indent 2), and `loss_curve.png` (epoch vs loss, Agg-safe), and returns the directory.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add pick_patch.py tests/test_pick_patch.py
git commit -m "feat: add patch metrics and run saving"
```

---

### Task 4: Selection, result figure, and CLI

**Files:**
- Modify: `pick_patch.py`
- Modify: `tests/test_pick_patch.py`

**Interfaces:**
- Consumes: every helper from Tasks 1–3.
- Produces:
  - `extents_to_patch(extents: tuple[float, float, float, float] | None, height: int, width: int, multiple: int = 8, min_size: int = 32) -> tuple[int, int, int, int] | None`
  - `validate_explicit_patch(x: int, y: int, w: int, h: int, height: int, width: int, multiple: int = 8) -> tuple[int, int, int, int]`
  - `validate_slice_index(index: int, n_slices: int) -> None`
  - `select_patch(slice_hu: np.ndarray, display_window: tuple[float, float]) -> tuple[int, int, int, int] | None`
  - `make_result_figure(*, slice_hu: np.ndarray, patch_input_hu: np.ndarray, patch_denoised_hu: np.ndarray, sigmas: list[np.ndarray], coords: tuple[int, int, int, int], metrics: dict, display_window: tuple[float, float], save_path: Path) -> Path`
  - `parse_args(argv: list[str] | None = None) -> argparse.Namespace`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_pick_patch.py


def test_extents_to_patch_reorders_and_handles_cancel():
    assert pick_patch.extents_to_patch((10, 74, 20, 90), 265, 370) == (10, 19, 64, 72)
    assert pick_patch.extents_to_patch(None, 265, 370) is None


def test_select_patch_requires_a_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError):
        pick_patch.select_patch(np.zeros((8, 8), dtype=np.float32), (-200, 400))


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: FAIL with `AttributeError: module 'pick_patch' has no attribute 'extents_to_patch'`.

- [ ] **Step 3: Implement selection, figure, and CLI**

`extents_to_patch` reorders matplotlib's `(xmin, xmax, ymin, ymax)` into `snap_patch`'s `(x0, y0, x1, y1)`; `None` means cancel.

`validate_explicit_patch` requires `w` and `h` to be positive multiples of `multiple` and the rectangle to fit inside `(height, width)`.

`validate_slice_index` raises `ValueError` unless `0 <= index < n_slices`.

`select_patch` first raises `RuntimeError("No graphical display: set DISPLAY or WAYLAND_DISPLAY")` when running on Linux with neither variable set. It then displays `slice_hu` with `imshow(..., cmap="gray", vmin=lo, vmax=hi)`, adds a `RectangleSelector(..., spancoords="pixels", interactive=True)`, sets the title "Drag a rectangle. Enter = confirm, Esc = cancel", and blocks on `plt.show()`. Confirm reads the selector extents; Esc or closing the window leaves them unset. Return `extents_to_patch(extents, ...)`.

`make_result_figure` lays out six panels: full slice with the rectangle, input patch, denoised patch, amplified difference, `sigma_r`, and the spatial sigma magnitude, with metrics in the title. Save to `result.png` under the run directory and close the figure.

`parse_args` defines every flag in Global Constraints plus `--patch X Y W H` (`nargs=4`, `type=int`, default `None`).

`main` loads the volume with `np.load(..., mmap_mode="r")`, validates the slice, selects or validates the patch, crops in HU, converts to unit, pads to 16, trains, crops the output back, computes metrics, builds the config dict, saves the run, writes the figure, prints a one-line summary, and returns 0. Esc prints `cancelled` and returns 0.

The config dict has exactly these keys: `input`, `slice`, `coords`, `window`, `display_window`, `epochs`, `lr`, `lambda_weight`, `num_stages`, `patch_size`, `seed`, `device`, `torch_version`, `params`, `train_seconds`, `loss`, `metrics`, `losses`, `weights`, `trained_on`. The last two are the strings `"random_init"` and `"selected_patch_only"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add pick_patch.py tests/test_pick_patch.py
git commit -m "feat: add patch picker UI and CLI"
```

---

### Task 5: End-to-end verification on slice 106

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes: the complete `pick_patch.py`.
- Produces: a verified run under `runs/patch_test/`.

- [ ] **Step 1: Run the full test suite**

Run: `.venv_cpu/bin/python -m pytest tests/test_pick_patch.py -v`
Expected: 14 passed.

- [ ] **Step 2: Run the non-GUI path on the real volume**

Run: `.venv_cpu/bin/python pick_patch.py --slice 106 --patch 120 140 64 64 --epochs 500`
Expected: "trained ... params=3638", a run directory containing the four files, finite metrics, and a `result.png`.

- [ ] **Step 3: Check the saved config**

Run: `.venv_cpu/bin/python -c "import json,glob; p=glob.glob('runs/patch_test/*/run_config.json')[0]; c=json.load(open(p)); print(c['coords'], c['weights'], c['trained_on'], c['metrics'])"`
Expected: `(120, 140, 64, 64)` (or the snapped list), `random_init`, `selected_patch_only`, finite metric values.

- [ ] **Step 4: Run the interactive path manually**

Run: `.venv_cpu/bin/python pick_patch.py --slice 106`
Drag a rectangle, press Enter, confirm the result figure opens. Then run again and press Esc; confirm it prints `cancelled` and writes nothing.

- [ ] **Step 5: Commit any fixes**

```bash
git add pick_patch.py tests/test_pick_patch.py
git commit -m "fix: address end-to-end picker issues"
```
