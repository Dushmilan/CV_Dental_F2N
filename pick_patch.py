"""Pick a CBCT patch interactively and denoise it with a fresh Filter2Noise model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from filter2noise import DenoisingPipeline, LossFunction


def snap_patch(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    height: int,
    width: int,
    multiple: int = 8,
    min_size: int = 32,
) -> tuple[int, int, int, int]:
    """Turn a dragged rectangle into a valid ``(x, y, w, h)`` patch."""
    if height < multiple or width < multiple:
        raise ValueError(
            f"Image {width}x{height} is smaller than the {multiple}-pixel patch grid."
        )
    if multiple <= 0 or min_size <= 0:
        raise ValueError("Patch multiple and minimum size must be positive.")
    min_size = math.ceil(min_size / multiple) * multiple
    max_w = (width // multiple) * multiple
    max_h = (height // multiple) * multiple
    if max_w < min_size or max_h < min_size:
        raise ValueError(
            f"Image {width}x{height} cannot fit the {min_size}x{min_size} minimum patch."
        )

    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    xa = min(max(xa, 0.0), float(width))
    xb = min(max(xb, 0.0), float(width))
    ya = min(max(ya, 0.0), float(height))
    yb = min(max(yb, 0.0), float(height))

    w = max(min_size, int(round((xb - xa) / multiple)) * multiple)
    h = max(min_size, int(round((yb - ya) / multiple)) * multiple)
    w = min(w, max_w)
    h = min(h, max_h)

    cx = (xa + xb) / 2.0
    cy = (ya + yb) / 2.0
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    x = min(max(x, 0), width - w)
    y = min(max(y, 0), height - h)
    return x, y, w, h


def hu_to_unit(image_hu: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Clip and normalise HU values to ``[0, 1]`` over ``window``."""
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        raise ValueError(f"Invalid HU window: {window}")
    scaled = (image_hu.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def unit_to_hu(image_unit: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Convert ``[0, 1]`` values back to HU over ``window``."""
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        raise ValueError(f"Invalid HU window: {window}")
    return (image_unit.astype(np.float32) * (hi - lo) + lo).astype(np.float32)


def pad_to_multiple(image: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad a 2D image to a multiple of ``multiple``, roughly centred."""
    h, w = image.shape
    target_h = math.ceil(h / multiple) * multiple
    target_w = math.ceil(w / multiple) * multiple
    pad_h = target_h - h
    pad_w = target_w - w
    top = pad_h // 2
    left = pad_w // 2
    padded = np.pad(
        image,
        ((top, pad_h - top), (left, pad_w - left)),
        mode="reflect",
    )
    return padded, (top, left)


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` or a device name to a ``torch.device``."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_patch(
    patch_unit: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_weight: float,
    num_stages: int = 2,
    patch_size: int = 8,
    seed: int = 77,
    log_every: int = 50,
) -> tuple[np.ndarray, list[float], dict]:
    """Train a fresh Filter2Noise model on one patch and return the result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tensor = torch.from_numpy(patch_unit.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    model = DenoisingPipeline(num_stages=num_stages, patch_size=patch_size).to(device)
    with torch.no_grad():
        model(tensor)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_function = LossFunction(device, lambda_=lambda_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=1
    )

    model.train()
    started = time.time()
    losses: list[float] = []
    last_loss = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(tensor, model)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        losses.append(last_loss)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch + 1:4d}/{epochs}  loss={last_loss:.6f}")

    train_seconds = time.time() - started
    model.eval()
    with torch.no_grad():
        denoised, sigmas = model(tensor, return_sigmas=True)

    output = denoised.squeeze().detach().cpu().numpy().astype(np.float32)
    sigma_arrays = [s.detach().cpu().numpy()[0].astype(np.float32) for s in sigmas]
    info = {
        "loss": last_loss,
        "train_seconds": round(train_seconds, 2),
        "params": int(params),
        "sigmas": sigma_arrays,
    }
    return output, losses, info


def compute_metrics(
    patch_hu: np.ndarray, denoised_hu: np.ndarray, window: tuple[float, float]
) -> dict:
    """Measure how far the denoised patch moved from the input patch."""
    lo, hi = float(window[0]), float(window[1])
    data_range = hi - lo
    diff = np.abs(
        denoised_hu.astype(np.float64) - patch_hu.astype(np.float64)
    )
    with np.errstate(divide="ignore"):
        psnr = peak_signal_noise_ratio(patch_hu, denoised_hu, data_range=data_range)
    ssim = structural_similarity(patch_hu, denoised_hu, data_range=data_range)
    return {
        "mean_abs_change_hu": float(diff.mean()),
        "max_abs_change_hu": float(diff.max()),
        "psnr_input": float(psnr),
        "ssim_input": float(ssim),
    }


def save_run(
    out_dir: Path,
    *,
    patch_input_hu: np.ndarray,
    patch_denoised_hu: np.ndarray,
    losses: list[float],
    config: dict,
) -> Path:
    """Write the patch arrays, config, and loss curve to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "patch_input_hu.npy", patch_input_hu)
    np.save(out / "patch_denoised_hu.npy", patch_denoised_hu)
    (out / "run_config.json").write_text(json.dumps(config, indent=2))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=150)
    plt.close(fig)
    return out


def extents_to_patch(
    extents: tuple[float, float, float, float] | None,
    height: int,
    width: int,
    multiple: int = 8,
    min_size: int = 32,
) -> tuple[int, int, int, int] | None:
    """Convert matplotlib ``(xmin, xmax, ymin, ymax)`` to ``(x, y, w, h)``."""
    if extents is None:
        return None
    x0, x1, y0, y1 = extents
    return snap_patch(x0, y0, x1, y1, height, width, multiple, min_size)


def validate_explicit_patch(
    x: int,
    y: int,
    w: int,
    h: int,
    height: int,
    width: int,
    multiple: int = 8,
    min_size: int = 32,
) -> tuple[int, int, int, int]:
    """Validate a patch supplied on the command line."""
    if w <= 0 or h <= 0:
        raise ValueError(f"Patch size must be positive, got {w}x{h}.")
    if w % multiple or h % multiple:
        raise ValueError(f"Patch size must be a multiple of {multiple}, got {w}x{h}.")
    if w < min_size or h < min_size:
        raise ValueError(
            f"Patch size must be at least {min_size}x{min_size}, got {w}x{h}."
        )
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(
            f"Patch ({x}, {y}, {w}, {h}) does not fit in {width}x{height}."
        )
    return int(x), int(y), int(w), int(h)


def validate_slice_index(index: int, n_slices: int) -> None:
    """Raise when ``index`` is outside the volume."""
    if not 0 <= index < n_slices:
        raise ValueError(f"Slice index {index} is outside 0..{n_slices - 1}.")


def select_patch(
    slice_hu: np.ndarray,
    display_window: tuple[float, float],
    *,
    multiple: int = 8,
    min_size: int = 32,
) -> tuple[int, int, int, int] | None:
    """Show the slice and return the rectangle the user confirms with Enter."""
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError("No graphical display: set DISPLAY or WAYLAND_DISPLAY")

    lo, hi = display_window
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_hu, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title("Drag a rectangle. Enter = confirm, Esc = cancel")
    ax.axis("off")

    state: dict = {"extents": None, "confirmed": False}

    def onselect(eclick, erelease) -> None:
        if None in (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata):
            return
        state["extents"] = (
            eclick.xdata,
            erelease.xdata,
            eclick.ydata,
            erelease.ydata,
        )

    # Keep a reference: matplotlib widgets are garbage-collected otherwise.
    selector = RectangleSelector(
        ax,
        onselect,
        useblit=False,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
    )
    selector.set_active(True)

    def on_key(event) -> None:
        if event.key == "enter":
            state["confirmed"] = True
            plt.close(fig)
        elif event.key in ("escape", "esc"):
            state["extents"] = None
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    plt.close(fig)

    if not state["confirmed"]:
        return None
    height, width = slice_hu.shape
    return extents_to_patch(state["extents"], height, width, multiple, min_size)


def make_result_figure(
    *,
    slice_hu: np.ndarray,
    patch_input_hu: np.ndarray,
    patch_denoised_hu: np.ndarray,
    sigmas: list[np.ndarray],
    coords: tuple[int, int, int, int],
    metrics: dict,
    display_window: tuple[float, float],
    save_path: Path,
    show: bool = False,
) -> Path:
    """Save a six-panel inspection figure for one patch run."""
    x, y, w, h = coords
    lo, hi = display_window
    diff = patch_denoised_hu.astype(np.float32) - patch_input_hu.astype(np.float32)
    diff_limit = max(1.0, float(np.abs(diff).max()) * 8.0)
    sigma_r = sigmas[0][..., 2]
    sigma_xy = np.sqrt(sigmas[0][..., 0] ** 2 + sigmas[0][..., 1] ** 2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (axes[0, 0], slice_hu, "gray", lo, hi, f"Slice, patch x={x} y={y} w={w} h={h}"),
        (axes[0, 1], patch_input_hu, "gray", lo, hi, "Patch input"),
        (axes[0, 2], patch_denoised_hu, "gray", lo, hi, "Patch denoised"),
        (axes[1, 0], diff * 8.0, "seismic", -diff_limit, diff_limit, "Difference x8 (HU)"),
        (axes[1, 1], sigma_r, "magma", None, None, "sigma_r (stage 1)"),
        (axes[1, 2], sigma_xy, "viridis", 0.0, None, "sqrt(sigma_x^2 + sigma_y^2)"),
    ]
    for ax, image, cmap, vmin, vmax, title in panels:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    axes[0, 0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1.5))
    fig.suptitle(
        f"mean|dHU|={metrics['mean_abs_change_hu']:.2f}  "
        f"max|dHU|={metrics['max_abs_change_hu']:.1f}  "
        f"PSNR={metrics['psnr_input']:.2f} dB  "
        f"SSIM={metrics['ssim_input']:.4f}"
    )
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/home/dushmilan/Desktop/data.npy")
    parser.add_argument("--slice", type=int, default=106)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-weight", type=float, default=200.0)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--window", type=float, nargs=2, default=(-1000.0, 2000.0))
    parser.add_argument("--display-window", type=float, nargs=2, default=(-200.0, 400.0))
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/patch_test")
    parser.add_argument("--min-patch", type=int, default=32)
    parser.add_argument(
        "--patch",
        nargs=4,
        type=int,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Skip the picker and use this patch.",
    )
    args = parser.parse_args(argv)
    if args.patch_size <= 0:
        parser.error("--patch-size must be positive")
    if args.min_patch < 32:
        parser.error("--min-patch must be at least 32")
    if args.min_patch % args.patch_size:
        parser.error("--min-patch must be a multiple of --patch-size")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run one interactive or explicit patch denoising job."""
    args = parse_args(argv)
    volume = np.load(args.input, mmap_mode="r")
    validate_slice_index(args.slice, volume.shape[0])
    slice_hu = np.asarray(volume[args.slice], dtype=np.float32)

    if args.patch is not None:
        coords = validate_explicit_patch(
            *args.patch,
            *slice_hu.shape,
            multiple=args.patch_size,
            min_size=args.min_patch,
        )
    else:
        coords = select_patch(
            slice_hu,
            tuple(args.display_window),
            multiple=args.patch_size,
            min_size=args.min_patch,
        )
        if coords is None:
            print("cancelled")
            return 0

    x, y, w, h = coords
    patch_hu = slice_hu[y:y + h, x:x + w]
    patch_unit = hu_to_unit(patch_hu, tuple(args.window))
    padded, (top, left) = pad_to_multiple(patch_unit, args.patch_size * 2)

    device = resolve_device(args.device)
    denoised_unit, losses, info = train_patch(
        padded,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_weight=args.lambda_weight,
        num_stages=args.num_stages,
        patch_size=args.patch_size,
        seed=args.seed,
    )

    denoised_padded = unit_to_hu(denoised_unit, tuple(args.window))
    denoised_hu = denoised_padded[top:top + h, left:left + w]
    cropped_sigmas = [
        sigma[top:top + h, left:left + w] for sigma in info["sigmas"]
    ]
    if not np.isfinite(denoised_unit).all() or not np.isfinite(denoised_hu).all():
        raise RuntimeError(f"Non-finite output for patch {coords}.")
    if not all(np.isfinite(sigma).all() for sigma in cropped_sigmas):
        raise RuntimeError(f"Non-finite sigma map for patch {coords}.")
    metrics = compute_metrics(patch_hu, denoised_hu, tuple(args.window))

    out_dir = Path(args.out) / f"slice_{args.slice}_x{x}_y{y}_w{w}_h{h}"
    config = {
        "input": str(Path(args.input).resolve()),
        "slice": args.slice,
        "coords": [x, y, w, h],
        "window": list(args.window),
        "display_window": list(args.display_window),
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_weight": args.lambda_weight,
        "num_stages": args.num_stages,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "params": info["params"],
        "train_seconds": info["train_seconds"],
        "loss": info["loss"],
        "metrics": metrics,
        "losses": losses,
        "weights": "random_init",
        "trained_on": "selected_patch_only",
    }
    save_run(
        out_dir,
        patch_input_hu=patch_hu,
        patch_denoised_hu=denoised_hu,
        losses=losses,
        config=config,
    )
    make_result_figure(
        slice_hu=slice_hu,
        patch_input_hu=patch_hu,
        patch_denoised_hu=denoised_hu,
        sigmas=cropped_sigmas,
        coords=coords,
        metrics=metrics,
        display_window=tuple(args.display_window),
        save_path=out_dir / "result.png",
        show=args.patch is None,
    )
    print(
        f"trained slice {args.slice} patch {coords}  params={info['params']}  "
        f"mean|dHU|={metrics['mean_abs_change_hu']:.2f}  -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
