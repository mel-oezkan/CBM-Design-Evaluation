"""Small, deterministic linear probes used by the leakage evaluator."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_logistic(x: torch.Tensor, y: torch.Tensor, n_classes: int, l2: float = 1e-3, steps: int = 200):
    """Multinomial logistic regression fitted with full-batch L-BFGS on standardized inputs."""
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    w = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    xs = (x - mean) / std

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xs @ w + b, y) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    w, b = w.detach(), b.detach()
    return lambda z: ((z - mean) / std) @ w + b


def balanced_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(1)
    accs = [float((pred[y == c] == c).float().mean()) for c in y.unique()]
    return sum(accs) / len(accs)
