"""Hugging Face Grounding DINO and SAM wrappers (``pip install cbm-eval[hf]``), loaded on first use.

Used by the ``grounding_dino`` concept scorer and the segment instance sources. Both classes take
PIL images and return boxes / masks in pixel coordinates of the original image.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    score: float
    label: int  # index into the image's prompt list, -1 if the phrase matched no prompt


def match_phrase(phrase: str, names: list[str]) -> int:
    """Map a detected phrase back to a prompt: exact match, else the longest prompt inside the
    phrase (Grounding DINO can merge adjacent phrases), else the one prompt containing it."""
    phrase = phrase.strip().lower()
    lowered = [n.strip().lower() for n in names]
    if not phrase:
        return -1
    if phrase in lowered:
        return lowered.index(phrase)
    inside = [i for i, n in enumerate(lowered) if n and n in phrase]
    if inside:
        return max(inside, key=lambda i: len(lowered[i]))
    containing = [i for i, n in enumerate(lowered) if phrase in n]
    return containing[0] if len(containing) == 1 else -1


class GroundingDINODetector:
    def __init__(self, model: str = "IDEA-Research/grounding-dino-tiny", box_threshold: float = 0.25,
                 text_threshold: float = 0.25, device: torch.device | str = "cpu"):
        self.model_id, self.box_threshold, self.text_threshold = model, box_threshold, text_threshold
        self.device = torch.device(device)
        self.model = self.processor = None

    def _load(self) -> None:
        if self.model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

    @torch.no_grad()
    def detect(self, images: list, prompts: list[list[str]]) -> list[list[Detection]]:
        """Boxes for each image, prompted with that image's own list of concept names."""
        self._load()
        texts = [". ".join(p.strip().lower() for p in ps) + "." for ps in prompts]
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        fn = self.processor.post_process_grounded_object_detection
        kw = {"text_threshold": self.text_threshold, "target_sizes": [im.size[::-1] for im in images]}
        # transformers renamed box_threshold -> threshold
        kw["threshold" if "threshold" in inspect.signature(fn).parameters else "box_threshold"] = self.box_threshold
        results = fn(outputs, inputs["input_ids"], **kw)
        out = []
        for res, names in zip(results, prompts):
            labels = res.get("text_labels", res.get("labels", []))
            out.append([Detection(tuple(float(v) for v in box), float(score), match_phrase(str(label), names))
                        for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), labels)])
        return out


class SAMSegmenter:
    def __init__(self, model: str = "facebook/sam-vit-base", device: torch.device | str = "cpu",
                 points_per_batch: int = 64):
        self.model_id, self.points_per_batch = model, points_per_batch
        self.device = torch.device(device)
        self.model = self.processor = None

    def _load(self) -> None:
        if self.model is None:
            from transformers import SamModel, SamProcessor

            self.processor = SamProcessor.from_pretrained(self.model_id)
            self.model = SamModel.from_pretrained(self.model_id).to(self.device).eval()

    def _decode(self, image, embeddings, multimask: bool, **prompts) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(image, return_tensors="pt", **prompts).to(self.device)
        inputs.pop("pixel_values")
        out = self.model(**inputs, image_embeddings=embeddings, multimask_output=multimask)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        return masks.bool(), out.iou_scores[0].cpu()  # (P, n_masks, H, W), (P, n_masks)

    def _embed(self, image) -> torch.Tensor:
        pixels = self.processor(image, return_tensors="pt")["pixel_values"].to(self.device)
        return self.model.get_image_embeddings(pixels)

    @torch.no_grad()
    def from_boxes(self, image, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
        """One (H, W) bool mask per box."""
        if not boxes:
            return []
        self._load()
        masks, _ = self._decode(image, self._embed(image), False, input_boxes=[[list(map(float, b)) for b in boxes]])
        return [m[0].numpy() for m in masks]

    @torch.no_grad()
    def from_points(self, image, points: list[tuple[float, float]]) -> tuple[list[np.ndarray], list[float]]:
        """The best of SAM's three candidate masks for each single-point prompt, with its predicted IoU."""
        self._load()
        emb = self._embed(image)
        masks, scores = [], []
        for start in range(0, len(points), self.points_per_batch):
            chunk = points[start : start + self.points_per_batch]
            m, iou = self._decode(image, emb, True, input_points=[[[list(map(float, p))] for p in chunk]],
                                  input_labels=[[[1] for _ in chunk]])
            best = iou.argmax(1)
            masks += [m[i, best[i]].numpy() for i in range(len(chunk))]
            scores += iou[torch.arange(len(chunk)), best].tolist()
        return masks, scores
