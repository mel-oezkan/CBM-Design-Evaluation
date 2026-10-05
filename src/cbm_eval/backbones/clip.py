"""OpenCLIP image/text encoders (``pip install cbm-eval[clip]``)."""

from __future__ import annotations

import torch

from ..registry import BACKBONES
from .base import Backbone


@BACKBONES.register("clip")
class CLIPBackbone(Backbone):
    has_text = True

    def __init__(self, model: str = "ViT-B-16", pretrained: str = "openai", template: str = "{}",
                 image_size: int = 224):
        super().__init__()
        import open_clip

        self.model_name, self.pretrained, self.template = model, pretrained, template
        self.model, _, self._preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model)
        self.model.eval()
        self.dim = self.model.text_projection.shape[1] if hasattr(self.model, "text_projection") \
            else self.model.visual.output_dim
        patch = getattr(self.model.visual, "patch_size", None)
        self.patch_grid = image_size // (patch[0] if isinstance(patch, tuple) else patch) if patch else None

    def cache_key(self):
        return {"name": "clip", "model": self.model_name, "pretrained": self.pretrained}

    def transform(self):
        return self._preprocess

    def to(self, device):
        self.model.to(device)
        return super().to(device)

    @torch.no_grad()
    def encode_images(self, images):
        return self.model.encode_image(images.to(self.device)).float()

    @torch.no_grad()
    def encode_patches(self, images):
        """Patch tokens projected into the joint space (after ``ln_post``), as in MaskCLIP-style maps."""
        visual = self.model.visual
        if not hasattr(visual, "output_tokens"):
            return super().encode_patches(images)
        visual.output_tokens = True
        try:
            _, tokens = visual(images.to(self.device))
        finally:
            visual.output_tokens = False
        if getattr(visual, "proj", None) is not None and tokens.shape[-1] != self.dim:
            tokens = tokens @ visual.proj
        return tokens.float()

    @torch.no_grad()
    def encode_text(self, texts):
        tokens = self.tokenizer([self.template.format(t) for t in texts]).to(self.device)
        return self.model.encode_text(tokens).float()
