"""Predictor: concept representation -> label. Sparse (GLM-SAGA style), dense, residual (PCBM-h),
or multiple-instance pooling over segments (SEG-MIL-CBM)."""

from __future__ import annotations

import torch
from torch import nn

from ..registry import PREDICTOR
from .base import Predictor, PredictorHead


class LinearHead(PredictorHead):
    uses_features = False

    def __init__(self, rep_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(rep_dim, num_classes)

    def forward(self, rep, x_norm, bag=None):
        return self.linear(rep)

    def stats(self):
        w = self.linear.weight
        return {"nonzero_frac": float((w.abs() > 1e-8).float().mean()),
                "nonzero_per_class": float((w.abs() > 1e-8).float().sum(1).mean())}


class SparseHead(LinearHead):
    """Elastic-net head fitted by proximal SGD: gradient step on loss + L2, then soft-threshold."""

    optimizer = "sgd"

    def __init__(self, rep_dim, num_classes, lam: float, alpha: float, lr: float):
        super().__init__(rep_dim, num_classes)
        self.lam, self.alpha, self.lr = lam, alpha, lr
        nn.init.zeros_(self.linear.weight)

    def penalty(self):
        return 0.5 * self.lam * (1 - self.alpha) * self.linear.weight.pow(2).sum()

    @torch.no_grad()
    def proximal_step(self):
        w = self.linear.weight
        t = self.lr * self.lam * self.alpha
        w.copy_(w.sign() * (w.abs() - t).clamp_min(0))


class ResidualHead(PredictorHead):
    """Interpretable linear head on concepts plus a linear residual on raw features."""

    def __init__(self, rep_dim, num_classes, feat_dim, residual_lam: float, sequential: bool):
        super().__init__()
        self.concept_head = nn.Linear(rep_dim, num_classes)
        self.residual = nn.Linear(feat_dim, num_classes)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.residual_lam, self.sequential = residual_lam, sequential

    def forward(self, rep, x_norm, bag=None):
        return self.concept_head(rep) + self.residual(x_norm)

    def phases(self):
        if self.sequential:  # PCBM-h: fit the concept head, then the residual on what it misses
            return [list(self.concept_head.parameters()), list(self.residual.parameters())]
        return [list(self.parameters())]

    def penalty(self):
        return self.residual_lam * self.residual.weight.pow(2).sum()

    @torch.no_grad()
    def concept_only(self, rep):
        return self.concept_head(rep)

    def stats(self):
        return {"residual_norm": float(self.residual.weight.detach().norm()),
                "concept_norm": float(self.concept_head.weight.detach().norm())}


@PREDICTOR.register("dense")
class Dense(Predictor):
    def build(self, rep_dim, num_classes, feat_dim):
        return LinearHead(rep_dim, num_classes)


@PREDICTOR.register("sparse")
class Sparse(Predictor):
    def __init__(self, lam: float = 1e-3, alpha: float = 0.99, lr: float = 0.1):
        self.lam, self.alpha, self.lr = lam, alpha, lr

    def build(self, rep_dim, num_classes, feat_dim):
        return SparseHead(rep_dim, num_classes, self.lam, self.alpha, self.lr)


@PREDICTOR.register("residual")
class Residual(Predictor):
    def __init__(self, residual_lam: float = 1e-3, sequential: bool = True):
        self.residual_lam, self.sequential = residual_lam, sequential

    def build(self, rep_dim, num_classes, feat_dim):
        return ResidualHead(rep_dim, num_classes, feat_dim, self.residual_lam, self.sequential)


class MILHead(PredictorHead):
    """Segment-level class evidence pooled over each image's instances (SEG-MIL-CBM).

    ``g_iy = eta_i * (w_y . z_i + b_y)`` is instance i's evidence for class y, with area weights
    ``eta_i = max(r_i, r_min)^-gamma`` normalized to mean 1 over the image's instances (all 1
    when ``area_gamma`` is 0). Attention pooling uses ``alpha_iy = softmax_i((a_y . tanh(A h_i + d)
    + e_y) / tau)`` on the normalized instance features h; ``mean`` and ``max`` are controls.
    The image logit is ``sum_i alpha_iy g_iy``. (N, D) inputs are treated as bags of one.
    """

    def __init__(self, rep_dim, num_classes, feat_dim, pooling: str, attn_dim: int, tau: float,
                 area_gamma: float, area_min: float):
        super().__init__()
        self.cls = nn.Linear(rep_dim, num_classes)
        self.pooling, self.tau, self.area_gamma, self.area_min = pooling, tau, area_gamma, area_min
        self.attn = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, num_classes)) \
            if pooling == "attention" else None

    def _area_weights(self, bag, mask):
        if not self.area_gamma or bag is None or bag.area is None:
            return mask.float()
        w = bag.area.clamp_min(self.area_min).pow(-self.area_gamma) * mask
        mean = w.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        return w / mean.clamp_min(1e-8)

    def _pool_weights(self, g, x_norm, mask):
        m = mask[..., None]
        if self.pooling == "attention":
            u = self.attn(x_norm) / self.tau
            return u.masked_fill(~m, -1e4).softmax(1) * m
        if self.pooling == "mean":
            w = m.to(g.dtype)
            return (w / w.sum(1, keepdim=True).clamp_min(1)).expand_as(g)
        top = g.detach().masked_fill(~m, float("-inf")).argmax(1, keepdim=True)
        return torch.zeros_like(g).scatter(1, top, 1.0) * m

    def evidence(self, rep, x_norm, bag=None):
        """Pooling weights alpha (N, M, C), area weights eta (N, M) and evidence g (N, M, C)."""
        if rep.dim() == 2:
            rep, x_norm, bag = rep[:, None], x_norm[:, None], None
        mask = bag.mask if bag is not None else torch.ones(rep.shape[:2], dtype=torch.bool, device=rep.device)
        eta = self._area_weights(bag, mask)
        g = eta[..., None] * self.cls(rep)
        return self._pool_weights(g, x_norm, mask), eta, g

    def forward(self, rep, x_norm, bag=None):
        alpha, _, g = self.evidence(rep, x_norm, bag)
        return (alpha * g).sum(1)

    def contributions(self, rep, x_norm, bag=None):
        """``C_iy = alpha_iy * eta_i * w_y . z_i``: each instance's share of the logit (bias excluded)."""
        alpha, eta, _ = self.evidence(rep, x_norm, bag)
        rep = rep[:, None] if rep.dim() == 2 else rep
        return alpha * eta[..., None] * (rep @ self.cls.weight.T)

    def stats(self):
        w = self.cls.weight.detach()
        return {"concept_norm": float(w.norm()), "nonzero_frac": float((w.abs() > 1e-8).float().mean())}


@PREDICTOR.register("mil")
class MIL(Predictor):
    supports_bags = True

    def __init__(self, pooling: str = "attention", attn_dim: int = 128, tau: float = 1.0,
                 area_gamma: float = 0.0, area_min: float = 0.01):
        if pooling not in ("attention", "mean", "max"):
            raise ValueError(f"Unknown MIL pooling '{pooling}' (use attention, mean or max)")
        self.pooling, self.attn_dim, self.tau = pooling, attn_dim, tau
        self.area_gamma, self.area_min = area_gamma, area_min

    def build(self, rep_dim, num_classes, feat_dim):
        return MILHead(rep_dim, num_classes, feat_dim, self.pooling, self.attn_dim, self.tau,
                       self.area_gamma, self.area_min)
