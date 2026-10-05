"""The trained CBM: concept layer + predictor head, plus cached concept scores per split."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .stages.base import AlignedConcepts, ConceptLayer, PredictorHead
from .structures import Bag, ConceptSet


class CBM(nn.Module):
    def __init__(self, layer: ConceptLayer, head: PredictorHead):
        super().__init__()
        self.layer, self.head = layer, head

    def forward(self, x: torch.Tensor, intervene=None, bag: Bag | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (class logits, concept predictions); concepts are per instance for bags."""
        c_hat, rep = self.layer(x, intervene)
        return self.head(rep, self.layer.normalize(x), bag), c_hat


@dataclass
class TrainedCBM:
    model: CBM
    aligned: AlignedConcepts
    config: dict[str, Any]
    train_log: dict[str, float] = field(default_factory=dict)
    concept_scores: dict[str, torch.Tensor] = field(default_factory=dict)
    encoder: nn.Module | None = None  # fine-tuned image encoder (training ``finetune:``); inputs are its features

    @property
    def concepts(self) -> ConceptSet:
        return self.aligned.concepts

    @torch.no_grad()
    def predict(self, x: torch.Tensor, intervene=None, bag: Bag | None = None,
                batch_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
        """(class logits, concept predictions), computed in chunks of ``batch_size`` images."""
        self.model.eval()
        logits, concepts = [], []
        for start in range(0, len(x), batch_size):
            sl = slice(start, start + batch_size)
            iv = None if intervene is None else (intervene[0][sl], intervene[1][sl])
            out = self.model(x[sl].float(), iv, None if bag is None else bag[sl])
            logits.append(out[0])
            concepts.append(out[1])
        return torch.cat(logits), torch.cat(concepts)

    def image_concepts(self, x: torch.Tensor, bag: Bag | None = None) -> torch.Tensor:
        """(N, K) concept predictions per image; instance bags are max-pooled over real instances."""
        c_hat = self.predict(x, bag=bag)[1]
        return bag.max(c_hat) if bag is not None and c_hat.dim() == 3 else c_hat

    def cache_scores(self, ctx, splits=("train", "val", "test")) -> None:
        for s in splits:
            self.concept_scores[s] = self.image_concepts(*ctx.inputs(s))

    def save(self, run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), run_dir / "model.pt")
        torch.save(self.concept_scores, run_dir / "concept_scores.pt")
        if self.encoder is not None:
            torch.save(self.encoder.state_dict(), run_dir / "encoder.pt")
        if self.concepts.vectors is not None:
            torch.save(self.concepts.vectors, run_dir / "concept_vectors.pt")
        (run_dir / "concepts.json").write_text(json.dumps(self.concepts.to_json(), indent=1))
        (run_dir / "config.json").write_text(json.dumps(self.config, indent=1, default=str))
        (run_dir / "train_log.json").write_text(json.dumps(self.train_log, indent=1))
        return run_dir
