"""Pick a CBCT patch interactively and denoise it with a fresh Filter2Noise model."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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

    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    xa = min(max(xa, 0.0), float(width))
    xb = min(max(xb, 0.0), float(width))
    ya = min(max(ya, 0.0), float(height))
    yb = min(max(yb, 0.0), float(height))

    w = max(min_size, int(round((xb - xa) / multiple)) * multiple)
    h = max(min_size, int(round((yb - ya) / multiple)) * multiple)
    w = min(w, (width // multiple) * multiple)
    h = min(h, (height // multiple) * multiple)

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
