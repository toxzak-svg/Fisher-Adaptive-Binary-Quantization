"""fisher_pass - compute per-channel Fisher Information on calibration data.

Pulled out of build_bucket.py so it's importable as a module function.
Used during the bucket-build step.
"""

from __future__ import annotations
import gc
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def fisher_pass(model, loader, max_batches: int = 16) -> dict:
    """Forward+backward over calibration data, accumulate gradient² per output channel.

    Returns: dict[layer_name, torch.Tensor of shape (out_features,)] on CPU.
    """
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "embed" in name.lower() or "lm_head" in name.lower():
            continue
        if "gate" in name.lower() or "router" in name.lower():
            continue
        module.register_buffer(
            "_fisher_buf",
            torch.zeros(module.out_features, device="cpu", dtype=torch.float32),
        )

        def _hook(mod, gi, go, m=module):
            if go[0] is None:
                return
            grad = go[0].detach().clone().to(torch.float32).cpu()
            if grad.dim() == 3:
                cf = (grad ** 2).sum(dim=[0, 1])
            else:
                cf = (grad ** 2).sum(dim=list(range(grad.dim() - 1)))
            if cf.shape[0] == m._fisher_buf.shape[0]:
                m._fisher_buf.add_(cf)
            del grad, cf

        h = module.register_full_backward_hook(_hook)
        hooks.append(h)

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    device = next(model.parameters()).device

    pbar = tqdm(loader, desc="Fisher", total=max_batches)
    for i, batch in enumerate(pbar):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(ids, labels=lbl)
                if out.loss is not None:
                    out.loss.backward()
                    model.zero_grad(set_to_none=True)
        except RuntimeError as e:
            print(f"  Batch {i}: {e}")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        del out, ids, lbl
        torch.cuda.empty_cache()
        gc.collect()

    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    for h in hooks:
        h.remove()

    return {n: m._fisher_buf.clone()
            for n, m in model.named_modules() if hasattr(m, "_fisher_buf")}
