"""Koh et al.'s fine-tuning augmentation on batches of decoded images, on the training device.

``train_transform`` augments one PIL image at a time in the dataloader workers, so fine-tuning
decodes and augments every JPEG again in every epoch, and the GPU waits for the CPU
(``reports/speedups.html``, P7). With finetune ``input: decoded`` the images are decoded once
(``Context.decoded``), and ``koh_augment`` applies the same augmentation to a whole batch with tensor
operations. It draws its parameters from the same distributions as ``train_transform``, but the
pixels differ:

- the crop is resampled by ``roi_align`` (bilinear, averaged over each output bin), not by PIL's resize;
- the colour jitter runs after the crop, not before it (both are per-pixel, so only the clamping
  at 0 and 1 can differ);
- brightness and saturation are applied in a random order, as ``ColorJitter`` does, and in float,
  not in 8-bit.

``roi_align`` needs one tensor, so the images are padded to a common size (``PaddedImages``), with
their right and bottom edge pixels repeated into the padding. Fine-tuning pads the whole training
set once (``pad_images``); a plain list is padded per batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Koh et al.'s augmentation (after Cui et al., 2018): ColorJitter(brightness, saturation), then
# RandomResizedCrop with torchvision's default scale and ratio. ``train_transform`` and ``koh_augment``
# both read these, so the PIL and the decoded path draw from the same distributions.
KOH_BRIGHTNESS, KOH_SATURATION = 32 / 255, (0.5, 1.5)
KOH_SCALE, KOH_RATIO = (0.08, 1.0), (3 / 4, 4 / 3)


@dataclass
class PaddedImages:
    """Decoded images of different sizes in one tensor: ``pixels`` (N, 3, H, W) uint8 with each image
    at the top left and its right / bottom edge pixels repeated into the padding; ``sizes`` (N, 2)
    their own (h, w), on the CPU."""

    pixels: torch.Tensor
    sizes: torch.Tensor

    def __len__(self) -> int:
        return len(self.sizes)


def pad_images(images: list[torch.Tensor], device: torch.device | None = None) -> PaddedImages:
    """uint8 images [(3, H_i, W_i)] -> ``PaddedImages`` of the largest height and width, on ``device``
    (default: the first image's)."""
    device = images[0].device if device is None else device
    sizes = torch.tensor([im.shape[1:] for im in images], dtype=torch.long)
    pixels = torch.empty(len(images), 3, int(sizes[:, 0].max()), int(sizes[:, 1].max()), dtype=torch.uint8,
                         device=device)
    for k, im in enumerate(images):
        h, w = im.shape[1:]
        pixels[k, :, :h, :w] = im.to(device)
        # roi_align samples up to half a pixel past a box's right / bottom edge: repeat the edge pixels
        # there, so a crop at the border reads the image's own edge (as clamping would), not padding
        pixels[k, :, :h, w:] = pixels[k, :, :h, w - 1:w]
        pixels[k, :, h:, :] = pixels[k, :, h - 1:h, :]
    return PaddedImages(pixels, sizes)


def _crop_box(h: int, w: int, scale: tuple[float, float], ratio: tuple[float, float],
              g: torch.Generator) -> tuple[int, int, int, int]:
    """(top, left, height, width) of a random crop, as ``RandomResizedCrop.get_params``."""
    area, log_ratio = h * w, (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target = area * torch.empty(1).uniform_(*scale, generator=g).item()
        aspect = math.exp(torch.empty(1).uniform_(*log_ratio, generator=g).item())
        cw, ch = round(math.sqrt(target * aspect)), round(math.sqrt(target / aspect))
        if 0 < cw <= w and 0 < ch <= h:
            top = int(torch.randint(0, h - ch + 1, (1,), generator=g))
            left = int(torch.randint(0, w - cw + 1, (1,), generator=g))
            return top, left, ch, cw
    if w / h < min(ratio):  # fallback: the central crop
        cw, ch = w, round(w / min(ratio))
    elif w / h > max(ratio):
        ch, cw = h, round(h * max(ratio))
    else:
        cw, ch = w, h
    return (h - ch) // 2, (w - cw) // 2, ch, cw


def _brightness(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return (x * f[:, None, None, None]).clamp(0, 1)


def _saturation(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])  # ITU-R 601-2 luma, as PIL's "L"
    f = f[:, None, None, None]
    return (f * x + (1 - f) * gray).clamp(0, 1)


def koh_augment(images: PaddedImages | list[torch.Tensor], size: int, mean, std, g: torch.Generator,
                brightness: float = KOH_BRIGHTNESS, saturation: tuple[float, float] = KOH_SATURATION,
                scale: tuple[float, float] = KOH_SCALE, ratio: tuple[float, float] = KOH_RATIO) -> torch.Tensor:
    """Decoded uint8 images (``PaddedImages``, or a list [(3, H_i, W_i)] padded here) -> (B, 3, size, size)
    normalized float inputs, on the images' device: ColorJitter(brightness, saturation),
    RandomResizedCrop(size, scale, ratio) and a horizontal flip, with every random draw taken from ``g``
    (a CPU generator)."""
    from torchvision.ops import roi_align

    if not isinstance(images, PaddedImages):
        images = pad_images(images)
    x, device, b = images.pixels.float(), images.pixels.device, len(images)
    boxes = torch.empty(b, 5)
    for k, (h, w) in enumerate(images.sizes.tolist()):  # the crop boxes stay inside each image
        top, left, ch, cw = _crop_box(h, w, scale, ratio, g)
        boxes[k] = torch.tensor([k, left, top, left + cw, top + ch], dtype=torch.float32)
    flip = (torch.rand(b, generator=g) < 0.5).float()
    bf = torch.empty(b).uniform_(max(0.0, 1 - brightness), 1 + brightness, generator=g)
    sf = torch.empty(b).uniform_(*saturation, generator=g)
    first = (torch.rand(b, generator=g) < 0.5).float()  # brightness before saturation
    # one pinned, non-blocking copy: a copy from pageable memory would wait for the GPU to finish its queue
    params = torch.cat([boxes, flip[:, None], bf[:, None], sf[:, None], first[:, None]], 1)  # (B, 9)
    if device.type == "cuda":
        params = params.pin_memory()
    params = params.to(device, non_blocking=True)
    boxes, flip, bf, sf = params[:, :5], params[:, 5] > 0.5, params[:, 6], params[:, 7]
    first = (params[:, 8] > 0.5)[:, None, None, None]
    x = roi_align(x, boxes, output_size=size, aligned=True) / 255  # (B, 3, size, size)
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x = torch.where(first, _saturation(_brightness(x, bf), sf), _brightness(_saturation(x, sf), bf))
    mean = torch.as_tensor(mean, dtype=x.dtype, device=device)[None, :, None, None]
    std = torch.as_tensor(std, dtype=x.dtype, device=device)[None, :, None, None]
    return (x - mean) / std
