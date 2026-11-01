"""Shared training loop: warm-up + cosine, early stopping on Val-A T1 AUROC, time cap, stop rule."""
import math
import time
from pathlib import Path

import numpy as np
import torch

from .metrics import t1_auroc, tz3_score
from .utils import autocast, predict_logits, save_json


def warmup_cosine(warmup_steps, total_steps, min_ratio=0.0):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


def fit(model, train_dl, val_dl, out_dir: Path, loss_fn, opt, dev, *, epochs, warmup_epochs, patience,
        accum=1, min_lr_ratio=0.0, time_cap_h=None, abort_rule=None):
    """Train `model`; keep the checkpoint with the best Val-A T1 AUROC in out_dir/model.pt.

    loss_fn(logits, y, x) -> scalar. abort_rule = (epoch, min_auroc): stop if, after that many
    epochs, the best Val-A AUROC is still below min_auroc.
    """
    steps_per_epoch = len(train_dl) // accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(warmup_epochs * steps_per_epoch, epochs * steps_per_epoch, min_lr_ratio))
    ctx, dtype = autocast(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    yA = val_dl.dataset.df.label.to_numpy()
    best, best_epoch, history, t0, status = -1.0, -1, [], time.time(), "completed"
    for epoch in range(epochs):
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(train_dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with ctx:
                loss = loss_fn(model(x).float(), y, x) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            losses.append(loss.item() * accum)
        auc = t1_auroc(yA, tz3_score(predict_logits(model, val_dl, dev)))
        minutes = (time.time() - t0) / 60
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), valA_auroc=auc, minutes=minutes))
        print(f"[{out_dir.name}] epoch {epoch:3d} loss {np.mean(losses):.4f} valA AUROC {auc:.4f} "
              f"({minutes:.1f} min)", flush=True)
        if auc > best:
            best, best_epoch = auc, epoch
            torch.save(model.state_dict(), out_dir / "model.pt")
        if epoch - best_epoch >= patience:
            status = f"early stop at epoch {epoch}"
            break
        if abort_rule and epoch + 1 == abort_rule[0] and best < abort_rule[1]:
            status = "aborted"
            break
        if time_cap_h and minutes / 60 >= time_cap_h:
            status = f"time cap at epoch {epoch}"
            break
    print(f"[{out_dir.name}] {status}; best valA AUROC {best:.4f} at epoch {best_epoch}")
    summary = dict(best_valA_auroc=best, best_epoch=best_epoch, minutes=(time.time() - t0) / 60,
                   status=status, history=history)
    save_json(out_dir / "summary.json", summary)
    return summary
