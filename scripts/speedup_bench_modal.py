"""GPU micro-benchmarks behind ``reports/speedups.html`` (the speed-up proposals), run on Modal.

Same image and GPU as ``koh2020_modal.py``. Measures, on CUB-sized inputs:

- ``finetune``: one Inception-v3 fine-tuning step (batch 64, 299 px, SGD) in fp32, channels_last,
  bf16 autocast, both, and ``torch.compile`` -- the inner loop of ``optimize_images``.
- ``loader``: CUB images through the backbone's ``train_transform`` with N workers, and the time
  to first batch (paid every epoch per loader without ``persistent_workers``).
- ``extract``: ``encode_patches`` + ``encode_images`` (what ``load_features`` does with patches)
  vs. one forward that returns both.
- ``head``: the sequential head phase with the concept layer recomputed per step vs. precomputed.
- ``profile``: a frozen-backbone run of the Koh et al. anchors on the cached CUB features, with the
  wall time split by pipeline step (cProfile cumulative times).

    uv run modal run scripts/speedup_bench_modal.py            # micro-benchmarks (A100)
    uv run modal run scripts/speedup_bench_modal.py::profile   # frozen CUB runs, time per step (L4)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

VOL = "/vol"
CUB_TGZ = f"{VOL}/data/CUB_200_2011.tgz"

app = modal.App("cbm-speedup-bench")
vol = modal.Volume.from_name("cbm-koh2020", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "numpy", "pandas", "pyyaml", "pillow")
    .add_local_python_source("cbm_eval")
    .add_local_dir(REPO / "configs", "/root/configs")
)


def _timed(fn, n: int, sync) -> float:
    """Mean seconds per call over ``n`` calls after two warm-up calls."""
    for _ in range(2):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t) / n


def bench_finetune(steps: int = 30) -> dict:
    import torch
    import torch.nn.functional as F

    from cbm_eval.backbones.inception import InceptionBackbone

    torch.manual_seed(0)
    x = torch.randn(64, 3, 299, 299, device="cuda")
    y = torch.randint(0, 112, (64, 112), device="cuda").float().clamp(max=1)
    out = {}
    for name, cl, amp, compile_ in [("fp32", False, False, False), ("channels_last", True, False, False),
                                    ("bf16", False, True, False), ("bf16+channels_last", True, True, False),
                                    ("bf16+channels_last+compile", True, True, True)]:
        torch.backends.cudnn.benchmark = True
        enc = InceptionBackbone().to(torch.device("cuda")).encoder().cuda().train()
        head = torch.nn.Linear(2048, 112).cuda()
        if cl:
            enc = enc.to(memory_format=torch.channels_last)
        inp = x.contiguous(memory_format=torch.channels_last) if cl else x
        model = torch.compile(enc) if compile_ else enc
        opt = torch.optim.SGD([*enc.parameters(), *head.parameters()], lr=1e-3, momentum=0.9)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.binary_cross_entropy_with_logits(head(model(inp)).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        t = _timed(step, steps, torch.cuda.synchronize)
        out[name] = {"s_per_step": t, "img_per_s": 64 / t}
        del enc, head, model, opt
        torch.cuda.empty_cache()
    return out


def bench_loader(root: str) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from cbm_eval.backbones.inception import InceptionBackbone
    from cbm_eval.data.cub import CUB

    t = time.perf_counter()
    ds = CUB(root=root)
    out = {"cub_init_s": time.perf_counter() - t}
    split = ds.torch_split("train", InceptionBackbone().train_transform())
    for workers in (8, 16):
        for persistent in (False, True):
            loader = DataLoader(split, batch_size=64, shuffle=True, drop_last=True, num_workers=workers,
                                pin_memory=True, persistent_workers=persistent,
                                generator=torch.Generator().manual_seed(0))
            firsts, totals = [], []
            for _ in range(3):  # three epochs: worker start-up is paid each epoch unless persistent
                t = time.perf_counter()
                it = iter(loader)
                next(it)
                firsts.append(time.perf_counter() - t)
                n = 1
                for _ in it:
                    n += 1
                    if n == 40:
                        break
                totals.append(40 * 64 / (time.perf_counter() - t))
            out[f"w{workers}{'_persistent' if persistent else ''}"] = {
                "first_batch_s_per_epoch": firsts, "img_per_s": sum(totals[1:]) / 2}
            del loader
    return out


def bench_extract(n: int = 10) -> dict:
    import torch

    from cbm_eval.backbones.inception import InceptionBackbone

    bb = InceptionBackbone().to(torch.device("cuda"))
    x = torch.randn(64, 3, 299, 299, device="cuda")
    sync = torch.cuda.synchronize
    two = _timed(lambda: (bb.encode_patches(x), bb.encode_images(x)), n, sync)
    one = _timed(lambda: (lambda p: (p.mean(1), p))(bb.encode_patches(x)), n, sync)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        amp = _timed(lambda: (lambda p: (p.mean(1), p))(bb.encode_patches(x)), n, sync)
    return {"patches_plus_global_s_per64": two, "single_pass_s_per64": one, "single_pass_bf16_s_per64": amp}


def bench_head(epochs: int = 20) -> dict:
    import torch
    import torch.nn.functional as F

    from cbm_eval.stages.base import AlignedConcepts
    from cbm_eval.stages.generation import ScoreLayer
    from cbm_eval.stages.predictor import SparseHead
    from cbm_eval.stages.training import Batch, OptimizerSpec, optimize
    from cbm_eval.structures import ConceptSet

    n, nv, d, k, c = 4796, 1198, 2048, 112, 200
    dev = torch.device("cuda")
    x, xv = torch.randn(n, d, device=dev), torch.randn(nv, d, device=dev)
    y, yv = torch.randint(0, c, (n,), device=dev), torch.randint(0, c, (nv,), device=dev)
    aligned = AlignedConcepts(ConceptSet([f"c{i}" for i in range(k)]), target_type="binary")

    def fit(precompute: bool) -> float:
        torch.manual_seed(0)
        layer = ScoreLayer(d, aligned).to(dev)
        layer.fit_input_stats(x)
        head = SparseHead(k, c, 1e-3, 0.99, 0.1).to(dev)
        if precompute:
            with torch.no_grad():
                tr = Batch(layer.activate(layer.concept_logits(x)), y)
                va = Batch(layer.activate(layer.concept_logits(xv)), yv)
            rep = lambda b: b.x
        else:
            tr, va = Batch(x, y), Batch(xv, yv)

            def rep(b):
                with torch.no_grad():
                    return layer.activate(layer.concept_logits(b.x))
        loss = lambda b: F.cross_entropy(head(rep(b), None), b.y)
        torch.cuda.synchronize()
        t = time.perf_counter()
        optimize([(list(head.parameters()), OptimizerSpec("sgd", 0.1))], [layer, head], n, lambda i: loss(tr[i]) + head.penalty(),
                 lambda: float(loss(va)), epochs, 256, 10**9, 0, after_step=head.proximal_step)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / epochs

    fit(False)  # warm-up
    return {"recompute_s_per_epoch": fit(False), "precomputed_s_per_epoch": fit(True)}


@app.function(image=image, volumes={VOL: vol}, gpu="A100-40GB", cpu=16, memory=65536, timeout=3600)
def run_all() -> dict:
    import tarfile

    import torch

    out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    out["head"] = bench_head()
    out["extract"] = bench_extract()
    out["finetune"] = bench_finetune()
    data = Path("/root/data")
    if Path(CUB_TGZ).exists():
        with tarfile.open(CUB_TGZ) as t:
            t.extractall(data)
        out["loader"] = bench_loader(str(data / "CUB_200_2011"))
    return out


def _step_timers() -> tuple[dict[str, float], list]:
    """Wrap pipeline steps with wall-clock accumulators (cProfile drops outer frames on 3.12 and
    inflates the many small training calls). Returns the totals and an undo list."""
    from cbm_eval import context, pipeline
    from cbm_eval.data.cub import CUB
    from cbm_eval.evaluation.concepts import ConceptEvaluator
    from cbm_eval.evaluation.leakage import LeakageEvaluator
    from cbm_eval.evaluation.shift import ShiftEvaluator
    from cbm_eval.model import TrainedCBM
    from cbm_eval.stages.training import _Base

    totals: dict[str, float] = {}
    undo = []
    targets = [("context", pipeline.PipelineBuilder, "context"), ("dataset init", CUB, "__init__"),
               ("feature load", context, "load_features"), ("concept phase", _Base, "_fit_concepts"),
               ("head phase", _Base, "_fit_head"), ("cache scores", TrainedCBM, "cache_scores"),
               ("eval: shift", ShiftEvaluator, "evaluate"), ("eval: concepts", ConceptEvaluator, "evaluate"),
               ("eval: leakage", LeakageEvaluator, "evaluate")]
    for label, owner, name in targets:
        fn = getattr(owner, name)
        totals[label] = 0.0

        def wrapped(*a, _fn=fn, _label=label, **kw):
            t = time.perf_counter()
            try:
                return _fn(*a, **kw)
            finally:
                totals[_label] += time.perf_counter() - t

        setattr(owner, name, wrapped)
        undo.append((owner, name, fn))
    return totals, undo


@app.function(image=image, volumes={VOL: vol}, gpu="L4", cpu=8, memory=32768, timeout=3600)
def profile_frozen(anchor: str, device: str) -> dict:
    """A frozen-backbone run of a Koh et al. anchor (as in its ablation), timed per pipeline step;
    then the same config again on the first run's Context (what reusing it in a sweep would save)."""
    import os
    import tarfile

    from cbm_eval import pipeline
    from cbm_eval.config import load_config

    work = Path("/root/work")
    (work / "data").mkdir(parents=True, exist_ok=True)
    with tarfile.open(CUB_TGZ) as t:
        t.extractall(work / "data")
    for link, target in {work / "data" / "CUB_processed": Path(VOL, "data", "CUB_processed"),
                         work / ".cache": Path(VOL, "cache")}.items():
        link.symlink_to(target)
    os.chdir(work)
    cfg = load_config(f"/root/configs/anchors/{anchor}.yaml", {
        "stages.training.finetune": None, "stages.training.device": device, "seed": 1,
        "paths": {"cache": ".cache", "runs": "/tmp/runs", "results": "/tmp/results.jsonl"}})
    out = {"anchor": anchor, "device": device}
    built = []
    context = pipeline.PipelineBuilder.context
    pipeline.PipelineBuilder.context = lambda self: built.append(context(self)) or built[-1]
    for name, ctx in (("fresh", None), ("reused_context", "reuse")):
        totals, undo = _step_timers()
        t = time.perf_counter()
        res = pipeline.run(cfg, save=False, ctx=built[0] if ctx else None)
        out[name] = {"total_s": time.perf_counter() - t, "steps_s": dict(totals),
                     "train_log": {k: v for k, v in res.cbm.train_log.items() if "epoch" in k},
                     "metrics": {k: v for k, v in res.metrics.items() if k.startswith(("shift.test.acc", "concepts.test"))}}
        for owner, attr, fn in undo:
            setattr(owner, attr, fn)
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="A100-40GB", cpu=16, memory=65536, timeout=3600)
def profile_finetune_epoch(epochs: int = 2) -> dict:
    """Epochs of the ``cub_koh2020`` fine-tuning loop on the anchor's own ``_ImageFeed`` loaders,
    split into time spent waiting for the next batch vs. computing, for train and val, under:
    the current code, bf16 + channels_last, and val images preprocessed once and kept in memory."""
    import os
    import tarfile

    import torch
    import torch.nn.functional as F

    from cbm_eval.config import load_config
    from cbm_eval.pipeline import PipelineBuilder
    from cbm_eval.stages.training import _ImageFeed

    work = Path("/root/work")
    (work / "data").mkdir(parents=True, exist_ok=True)
    with tarfile.open(CUB_TGZ) as t:
        t.extractall(work / "data")
    for link, target in {work / "data" / "CUB_processed": Path(VOL, "data", "CUB_processed"),
                         work / ".cache": Path(VOL, "cache")}.items():
        link.symlink_to(target)
    os.chdir(work)
    cfg = load_config("/root/configs/anchors/cub_koh2020.yaml", {"stages.training.device": "cuda", "seed": 1})
    builder = PipelineBuilder(cfg)
    ctx = builder.context()
    aligned = builder.alignment.fit(builder.concepts(ctx), ctx)
    dev = torch.device("cuda")
    spec = builder.training.finetune
    out = {"n_train": len(ctx.split("train")), "n_val": len(ctx.split("val")), "spec": {
        "batch_size": spec.batch_size, "num_workers": spec.num_workers}}

    for variant in ("current", "bf16+channels_last", "cached val"):
        torch.manual_seed(1)
        torch.backends.cudnn.benchmark = variant != "current"
        encoder = ctx.backbone.encoder().to(dev)
        layer = builder.generation.build(ctx.feat_dim, aligned).to(dev)
        head = builder.predictor.build(layer.rep_dim, ctx.num_classes, ctx.feat_dim).to(dev)
        fast = variant == "bf16+channels_last"
        fmt = torch.channels_last if fast else torch.contiguous_format
        encoder = encoder.to(memory_format=fmt)
        feed = _ImageFeed(ctx, aligned, encoder, spec, dev, 1)
        opt = spec.make_optimizer(encoder.parameters())
        opt2 = torch.optim.Adam([*layer.parameters(), *head.parameters()], lr=1e-3)
        val_cache = None
        if variant == "cached val":
            t = time.perf_counter()
            val_cache = [(item["image"].to(dev), item["index"]) for item in feed.loaders["val"]]
            out["val_cache_build_s"] = time.perf_counter() - t

        def loss_of(images, idx, split):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=fast):
                x = encoder(images.to(dev, non_blocking=True, memory_format=fmt)).float()
            c, rep = layer(x)
            y = feed.labels[split][idx].to(dev)
            return F.cross_entropy(head(rep, layer.normalize(x)), y) + \
                F.binary_cross_entropy_with_logits(layer.concept_logits(x), feed.targets[split][idx].to(dev))

        def timed_pass(batches, split, train):
            wait = compute = 0.0
            it = iter(batches)
            while True:
                t0 = time.perf_counter()
                try:
                    item = next(it)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                images, idx = (item["image"], item["index"]) if isinstance(item, dict) else item
                if train:
                    loss = loss_of(images, idx, split)
                    opt.zero_grad(); opt2.zero_grad()
                    loss.backward()
                    opt.step(); opt2.step()
                else:
                    with torch.no_grad():
                        loss_of(images, idx, split)
                torch.cuda.synchronize()
                wait, compute = wait + t1 - t0, compute + time.perf_counter() - t1
            return wait, compute

        per_epoch = []
        for _ in range(epochs):
            encoder.train(); layer.train(); head.train()
            tw, tc = timed_pass(feed.loaders["train"], "train", True)
            encoder.eval(); layer.eval(); head.eval()
            vw, vc = timed_pass(val_cache if val_cache else feed.loaders["val"], "val", False)
            per_epoch.append({"train_wait": tw, "train_compute": tc, "val_wait": vw, "val_compute": vc})
        out[variant] = per_epoch[-1]  # the last (warm) epoch
        del encoder, layer, head, opt, opt2, feed, val_cache
        torch.cuda.empty_cache()
    return out


def bench_frozen_gpu(epochs: int = 200, n: int = 4796, d: int = 2048, k: int = 112, bs: int = 256) -> dict:
    """The frozen concept phase (BCE, Adam, 200 epochs of 256-image minibatches) on the GPU: eager,
    fused Adam, the whole step captured in a CUDA graph, and nine seeds trained at once with bmm."""
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    g0 = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=g0).to(dev)
    t = (torch.rand(n, k, generator=g0) < 0.1).float().to(dev)
    steps = n // bs

    def perms(seed):
        g = torch.Generator().manual_seed(seed)
        return [torch.randperm(n, generator=g)[: steps * bs].view(steps, bs).to(dev) for _ in range(epochs)]

    order = perms(0)

    def eager(fused: bool):
        torch.manual_seed(0)
        lin = torch.nn.Linear(d, k).to(dev)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, fused=fused)
        torch.cuda.synchronize()
        s = time.perf_counter()
        for p in order:
            for idx in p:
                loss = F.binary_cross_entropy_with_logits(lin(x[idx]), t[idx])
                opt.zero_grad(set_to_none=False)
                loss.backward()
                opt.step()
        torch.cuda.synchronize()
        return time.perf_counter() - s, lin.weight.detach().clone()

    def graphed():
        torch.manual_seed(0)
        lin = torch.nn.Linear(d, k).to(dev)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, capturable=True)
        idx_buf = torch.zeros(bs, dtype=torch.long, device=dev)

        def step():
            loss = F.binary_cross_entropy_with_logits(lin(x[idx_buf]), t[idx_buf])
            opt.zero_grad(set_to_none=False)
            loss.backward()
            opt.step()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        state = {k_: v.clone() for k_, v in lin.state_dict().items()}
        with torch.cuda.stream(side):  # warm-up on a side stream, as capture requires
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        # restart from the same initial weights and a fresh optimizer state, then replay
        with torch.no_grad():
            lin.load_state_dict(state)
            for st in opt.state.values():
                st["exp_avg"].zero_(); st["exp_avg_sq"].zero_(); st["step"].zero_()
        torch.cuda.synchronize()
        s = time.perf_counter()
        for p in order:
            for idx in p:
                idx_buf.copy_(idx)
                graph.replay()
        torch.cuda.synchronize()
        return time.perf_counter() - s, lin.weight.detach().clone()

    def seeds(S: int):
        orders = [perms(s_) for s_ in range(S)]
        torch.manual_seed(0)
        W = torch.nn.Parameter(torch.randn(S, d, k, device=dev) / d**0.5)
        b = torch.nn.Parameter(torch.zeros(S, 1, k, device=dev))
        opt = torch.optim.Adam([W, b], lr=1e-3, fused=True)
        torch.cuda.synchronize()
        s = time.perf_counter()
        for e in range(epochs):
            for j in range(steps):
                idx = torch.stack([orders[s_][e][j] for s_ in range(S)])  # (S, B)
                loss = F.binary_cross_entropy_with_logits(torch.bmm(x[idx], W) + b, t[idx],
                                                          reduction="none").mean((1, 2)).sum()
                opt.zero_grad(set_to_none=False)
                loss.backward()
                opt.step()
        torch.cuda.synchronize()
        return time.perf_counter() - s

    eager(False)  # warm-up
    te, we = eager(False)
    tf_, wf = eager(True)
    tg, wg = graphed()
    return {"eager_s": te, "fused_adam_s": tf_, "cuda_graph_s": tg,
            "fused_maxdiff": float((we - wf).abs().max()), "graph_maxdiff": float((we - wg).abs().max()),
            "nine_seeds_batched_s": seeds(9), "nine_seeds_serial_estimate_s": 9 * tf_}


def bench_decoded_cache(root: str, batches: int = 60) -> dict:
    """Fine-tuning loader throughput with JPEGs decoded per epoch (current) vs. decoded once and kept as
    uint8 arrays shared with the workers, on the same seed: steady-state images/s and output equality."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from cbm_eval.backbones.inception import InceptionBackbone
    from cbm_eval.data.cub import CUB

    ds = CUB(root=root)
    tf = InceptionBackbone().train_transform()
    current = ds.torch_split("train", tf)
    samples = ds.samples("train")
    t = time.perf_counter()
    arrays = [np.asarray(Image.open(s.image).convert("RGB")) for s in samples]
    build = time.perf_counter() - t

    class Decoded(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            return {"image": tf(Image.fromarray(arrays[i])), "index": i}

    out = {"decode_all_s": build, "decoded_gb": sum(a.nbytes for a in arrays) / 1e9}
    first = {}
    for name, dset in (("current", current), ("decoded_cache", Decoded())):
        loader = DataLoader(dset, batch_size=64, shuffle=True, drop_last=True, num_workers=8, pin_memory=True,
                            generator=torch.Generator().manual_seed(0))
        it = iter(loader)
        t0 = time.perf_counter()
        b = next(it)
        t1 = time.perf_counter()
        first[name] = b["image"][:4].clone()
        for _ in range(batches - 1):
            next(it)
        out[name] = {"first_batch_s": t1 - t0, "steady_img_per_s": (batches - 1) * 64 / (time.perf_counter() - t1)}
        del it, loader
    out["identical_first_batch"] = bool(torch.equal(first["current"], first["decoded_cache"]))
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="A100-40GB", cpu=16, memory=65536, timeout=3600)
def training_extras() -> dict:
    import tarfile

    import torch

    out = {"gpu": torch.cuda.get_device_name(), "frozen": bench_frozen_gpu()}
    data = Path("/root/data")
    with tarfile.open(CUB_TGZ) as t:
        t.extractall(data)
    out["loader"] = bench_decoded_cache(str(data / "CUB_200_2011"))
    return out


@app.local_entrypoint()
def training():
    print(json.dumps(training_extras.remote(), indent=1))


@app.local_entrypoint()
def epoch():
    print(json.dumps(profile_finetune_epoch.remote(), indent=1))


@app.local_entrypoint()
def main():
    print(json.dumps(run_all.remote(), indent=1))


@app.local_entrypoint()
def profile():
    jobs = [(a, d) for a in ("cub_koh2020_independent", "cub_koh2020_sequential") for d in ("cpu", "cuda")]
    for r in profile_frozen.starmap(jobs):
        print(json.dumps(r, indent=1))
