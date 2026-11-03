# Patch-level Filter2Noise validation

**Date:** 2026-11-03
**Status:** draft, awaiting approval

## Intent

The whole-slice result at `lambda=350` changes only ~2.8 HU on average, which
looks like a near-identity mapping. That makes it impossible to tell, by eye,
whether Filter2Noise is actually removing noise or simply passing the input
through. The user also wants to confirm the "zero-shot" claim directly.

The tool lets the user look at a rendered slice, drag a rectangle around a
region they can judge visually, and train a **fresh** Filter2Noise model using
**only that patch**. The result is shown at patch scale, where noise texture is
visible.

## Goals

- Open a local matplotlib window showing one axial slice with the HU display
  window used in the existing runs (`[-200, 400]`).
- Let the user drag a rectangle and confirm with Enter (Esc cancels).
- Snap the rectangle to a multiple of 8, clamp it to the slice, minimum 32x32.
- Train a new `DenoisingPipeline(num_stages=2, patch_size=8)` from random
  initialisation on the selected patch alone, with the existing ELS /
  multi-scale loss.
- Show patch input, denoised output, difference, and predicted sigma maps in a
  result figure.
- Save the patch arrays, result figure, and a run config that records the exact
  coordinates and parameters.
- Report `mean|dHU|`, `max|dHU|`, PSNR/SSIM against the input patch, final loss,
  and wall time.

## Non-goals

- No changes to `run_cbct.py` or `filter2noise.py`.
- No full-slice training and no comparison against the full-slice model in the
  same run.
- No Gradio or browser UI.
- No batch mode and no clinical-quality claim.
- Not a substitute for a clean reference; the metrics still compare against the
  noisy patch.

## Constraints

- `SigmaPredictor` needs dimensions divisible by `patch_size` (8); `ELS` needs
  even dimensions; `Downsampler` halves twice. Dimensions that are multiples of
  16 are safe, matching `run_cbct.py`'s padding.
- `run_cbct.py` forces the matplotlib `Agg` backend at import time, so this
  script must not import it. It will import only `filter2noise` and duplicate
  the small HU/padding helpers.
- The local GUI works here: TkAgg, `DISPLAY=:1`, Wayland session.
- The volume is loaded with `mmap_mode="r"` so only the selected slice is read.
- CPU is the expected device; a 64x64 patch at 500 epochs should finish in a
  few seconds.

## Approaches considered

1. **One script, drag a rectangle (chosen).** Direct control of the region, one
   command, coordinates saved for reproducibility.
2. **Two-step pick-then-run CLI.** More scriptable, but adds a step and slows
   visual iteration.
3. **Click a centre point with a fixed patch size.** Fewer mis-drags, but less
   control over irregular anatomy.

## Design

New file: `pick_patch.py`.

### CLI

```
python pick_patch.py --slice 106
```

Defaults mirror the CBCT pilot: `--input /home/dushmilan/Desktop/data.npy`,
`--epochs 500`, `--lr 1e-3`, `--lambda-weight 350`, `--num-stages 2`,
`--window -1000 2000`, `--display-window -200 400`, `--seed 77`, `--device auto`,
`--out runs/patch_test`, `--min-patch 32`.

`--patch X Y W H` bypasses the GUI for exact reproduction of a saved run.

### Selection flow

1. Load the slice, convert to HU, display with the display window.
2. `RectangleSelector` with `spancoords="pixels"`; title says "Drag a rectangle.
   Enter = confirm, Esc = cancel".
3. On Enter: read extents, round to ints, clamp to image bounds, snap width and
   height to a multiple of 8, enforce the 32-pixel minimum by growing around the
   centre, re-clamp.
4. On Esc or window close without a selection: print "cancelled", exit 0.

Coordinates are saved as `x, y, w, h` with `x` = column and `y` = row of the
top-left corner.

### Training

- Crop the patch in HU, then `hu_to_unit` with `--window`, matching the existing
  normalisation.
- Reflect-pad to a multiple of 16, remembering the crop box.
- Build a fresh `DenoisingPipeline`, `AdamW(weight_decay=0.01)`, `OneCycleLR`
  with `max_lr=lr` and `steps_per_epoch=1`, batch size 1.
- Each step: forward the padded patch, compute `LossFunction(..., lambda_)`,
  backward, step.
- Crop the padded output back to the patch size and convert to HU.
- Print loss every 50 epochs.

### Result figure

Six panels: full slice with the selected rectangle, input patch, denoised patch,
difference (amplified), `sigma_x`/`sigma_y`, and `sigma_r` for stage 1. The title
includes slice, coordinates, `mean|dHU|`, `max|dHU|`, PSNR/SSIM vs input, final
loss, and seconds.

### Outputs

```
runs/patch_test/slice_106_x120_y140_w64_h64/
  patch_input_hu.npy
  patch_denoised_hu.npy
  patch_difference_hu.npy
  result.png
  loss_curve.png
  run_config.json
```

`run_config.json` records the input path, slice index, coordinates, all
parameters, `weights: random_init`, `trained_on: selected_patch_only`, device,
torch version, and the per-step loss list.

### Failure handling

- No selection / Esc: clean exit, no files written.
- Rectangle smaller than the minimum after snapping: clear error, no training.
- Non-finite loss or output: abort with the step number, keep no output files.
- Missing display: explain that a local GUI is required and suggest checking
  `DISPLAY`.

## What is left to the builder

- Exact figure layout and colormap details.
- Whether to show training progress in the console, a matplotlib status line, or
  both.
- File naming details for repeated runs at the same coordinates (overwrite vs
  suffix). Default: overwrite, since the config makes the run reproducible.

## Acceptance

- `python pick_patch.py --slice 106` opens the slice, a dragged 64x64 region
  trains, and the result figure appears.
- The saved config contains the coordinates and `weights: random_init`.
- Re-running with `--patch` and the same seed produces the same `mean|dHU|` to
  within floating-point noise.
- No existing source files are modified.
