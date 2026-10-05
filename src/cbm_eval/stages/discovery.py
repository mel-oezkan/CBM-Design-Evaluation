"""Discovery: propose candidate concepts (LLM, VLM, knowledge base, SAE, or a fixed list)."""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import DISCOVERY
from ..structures import ConceptSet
from ..utils import JsonCache
from .base import Discovery
from .llm import ClaudeClient, parse_list


def _merge(per_class: dict[str, list[str]], source: str) -> ConceptSet:
    """Union of per-class lists, remembering which classes proposed each concept."""
    origin: dict[str, list[str]] = {}
    for cls, items in per_class.items():
        for item in items:
            origin.setdefault(item, [])
            if cls not in origin[item]:
                origin[item].append(cls)
    names = list(origin)
    return ConceptSet(names=names, meta=[{"source": source, "classes": origin[n]} for n in names])


@DISCOVERY.register("static")
class StaticDiscovery(Discovery):
    """Concepts from the config (``names``) or a file (one per line, or a JSON list / {class: [..]} map)."""

    def __init__(self, names: list[str] | None = None, path: str | None = None):
        if (names is None) == (path is None):
            raise ValueError("static discovery needs exactly one of `names` or `path`")
        self.names, self.path = names, path

    def discover(self, ctx):
        if self.names is not None:
            names = list(dict.fromkeys(self.names))
            return ConceptSet(names=names, meta=[{"source": "static"} for _ in names])
        text = Path(self.path).read_text()
        if self.path.endswith(".json"):
            data = json.loads(text)
            if isinstance(data, dict):
                return _merge(data, "static")
            return ConceptSet(names=list(dict.fromkeys(data)))
        return ConceptSet(names=list(dict.fromkeys(l.strip() for l in text.splitlines() if l.strip())))


@DISCOVERY.register("dataset")
class DatasetDiscovery(Discovery):
    """The dataset's human-annotated concept vocabulary (required for human alignment)."""

    def __init__(self, extra: list[str] | None = None):
        self.extra = extra or []

    def discover(self, ctx):
        names = ctx.dataset.concept_names
        if not names:
            raise RuntimeError(f"Dataset '{ctx.dataset.name}' has no annotated concepts")
        names = list(dict.fromkeys(names + self.extra))
        return ConceptSet(names=names, meta=[{"source": "dataset"} for _ in names])


LLM_PROMPTS = {
    "features": "List the most important visual features for recognizing a \"{cls}\" in a photo.",
    "around": "List the things most commonly seen around a \"{cls}\" in a photo.",
    "superclass": "Give superclasses for the word \"{cls}\".",
}
_LIST_SUFFIX = ("\nAnswer with a plain list, one short noun phrase (1-4 words) per line, no numbering, "
                "no explanations, at most {n} items.")


@DISCOVERY.register("llm")
class LLMDiscovery(Discovery):
    """Per-class prompting in the style of Label-free CBM / LaBo."""

    def __init__(self, prompts: list[str] | None = None, per_class: int = 10, model: str | None = None,
                 effort: str = "medium", client=None):
        self.prompts = prompts or ["features", "around", "superclass"]
        self.per_class, self.model, self.effort, self.client = per_class, model, effort, client

    def _client(self, ctx):
        if self.client is None:
            kw = {"model": self.model} if self.model else {}
            self.client = ClaudeClient(effort=self.effort, cache_dir=ctx.cache_dir / "llm", **kw)
        return self.client

    def discover(self, ctx):
        client = self._client(ctx)
        per_class = {}
        for cls in ctx.class_names:
            items: list[str] = []
            for p in self.prompts:
                prompt = LLM_PROMPTS.get(p, p).format(cls=cls) + _LIST_SUFFIX.format(n=self.per_class)
                items += parse_list(client.complete(prompt))[: self.per_class]
            per_class[cls] = items
        return _merge(per_class, "llm")


@DISCOVERY.register("vlm")
class VLMDiscovery(Discovery):
    """Show a VLM a few training images per class and ask for the visible attributes."""

    PROMPT = ("These images all show a \"{cls}\". List the visual attributes and parts that are visible and "
              "would help recognize it." + _LIST_SUFFIX)

    def __init__(self, images_per_class: int = 4, per_class: int = 15, model: str | None = None,
                 effort: str = "medium", client=None):
        self.images_per_class, self.per_class = images_per_class, per_class
        self.model, self.effort, self.client = model, effort, client

    def discover(self, ctx):
        if self.client is None:
            kw = {"model": self.model} if self.model else {}
            self.client = ClaudeClient(effort=self.effort, cache_dir=ctx.cache_dir / "vlm", **kw)
        train = ctx.split("train")
        if not train.paths or train.paths[0] is None:
            raise RuntimeError("VLM discovery needs a dataset with image files")
        rng = random.Random(ctx.seed)
        per_class = {}
        for c, cls in enumerate(ctx.class_names):
            idx = (train.labels == c).nonzero().flatten().tolist()
            paths = [train.paths[i] for i in rng.sample(idx, min(self.images_per_class, len(idx)))]
            text = self.client.complete(self.PROMPT.format(cls=cls, n=self.per_class), images=paths)
            per_class[cls] = parse_list(text)[: self.per_class]
        return _merge(per_class, "vlm")


@DISCOVERY.register("kb")
class KnowledgeBaseDiscovery(Discovery):
    """ConceptNet edges (HasA, PartOf, HasProperty, AtLocation, ...) or a local {class: [concepts]} JSON."""

    def __init__(self, path: str | None = None, relations: list[str] | None = None, per_class: int = 20,
                 min_weight: float = 1.0, api: str = "https://api.conceptnet.io"):
        self.path, self.per_class, self.min_weight, self.api = path, per_class, min_weight, api
        self.relations = relations or ["HasA", "PartOf", "HasProperty", "MadeOf", "AtLocation", "IsA"]

    def discover(self, ctx):
        if self.path:
            data = json.loads(Path(self.path).read_text())
            return _merge({c: data.get(c, [])[: self.per_class] for c in ctx.class_names}, "kb")
        cache = JsonCache(ctx.cache_dir / "conceptnet")
        return _merge({c: self._query(c, cache)[: self.per_class] for c in ctx.class_names}, "kb")

    def _query(self, cls: str, cache: JsonCache) -> list[str]:
        term = cls.lower().replace(" ", "_")
        scored: dict[str, float] = {}
        for rel in self.relations:
            for direction in ("start", "end"):
                url = f"{self.api}/query?{direction}=/c/en/{urllib.parse.quote(term)}&rel=/r/{rel}&limit=100"
                edges = cache.get(url)
                if edges is None:
                    with urllib.request.urlopen(url, timeout=30) as r:
                        edges = json.load(r).get("edges", [])
                    cache.set(url, edges)
                other = "end" if direction == "start" else "start"
                for e in edges:
                    node = e[other]
                    if node.get("language") == "en" and e.get("weight", 0) >= self.min_weight:
                        label = node["label"].lower()
                        scored[label] = max(scored.get(label, 0.0), e["weight"])
        return sorted(scored, key=scored.get, reverse=True)


@DISCOVERY.register("sae")
class SAEDiscovery(Discovery):
    """TopK sparse autoencoder (Gao et al., 2024) on standardized training features.

    Each latent that fires on a reasonable fraction of images becomes a concept whose
    vector is its decoder direction. TopK fixes sparsity exactly, so there is no L1 to tune.
    """

    def __init__(self, n_latents: int = 256, k: int = 8, epochs: int = 50, lr: float = 1e-3,
                 batch_size: int = 256, min_frequency: float = 0.005, max_frequency: float = 0.5):
        self.n_latents, self.k, self.epochs, self.lr, self.batch_size = n_latents, k, epochs, lr, batch_size
        self.min_frequency, self.max_frequency = min_frequency, max_frequency

    def _encode(self, enc: nn.Linear, x: torch.Tensor, b_dec: torch.Tensor) -> torch.Tensor:
        pre = F.relu(enc(x - b_dec))
        top = pre.topk(self.k, dim=1)
        return torch.zeros_like(pre).scatter(1, top.indices, top.values)

    def discover(self, ctx):
        x = ctx.split("train").features.float()
        x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
        d = x.shape[1]
        g = torch.Generator().manual_seed(ctx.seed)
        enc = nn.Linear(d, self.n_latents)
        dec = nn.Parameter(F.normalize(torch.randn(self.n_latents, d, generator=g), dim=1))
        with torch.no_grad():
            enc.weight.copy_(dec)
            enc.bias.zero_()
        b_dec = nn.Parameter(torch.zeros(d))
        opt = torch.optim.Adam([*enc.parameters(), dec, b_dec], lr=self.lr)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(x), generator=g).split(self.batch_size):
                recon = self._encode(enc, x[idx], b_dec) @ dec + b_dec
                loss = (recon - x[idx]).pow(2).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    dec.copy_(F.normalize(dec, dim=1))
        with torch.no_grad():
            freq = (self._encode(enc, x, b_dec) > 0).float().mean(0)
        live = ((freq >= self.min_frequency) & (freq <= self.max_frequency)).nonzero().flatten().tolist()
        if not live:
            raise RuntimeError(f"SAE produced no latents with frequency in [{self.min_frequency}, "
                               f"{self.max_frequency}]; adjust k or n_latents")
        return ConceptSet(names=[f"sae_{i}" for i in live], vectors=dec.detach()[live].clone(),
                          meta=[{"source": "sae", "frequency": float(freq[i])} for i in live])
