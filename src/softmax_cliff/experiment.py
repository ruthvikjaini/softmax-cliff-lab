"""Training experiments: grokking at the precision cliff and distillation.

* :func:`train_grokking` — one full-batch AdamW run on a modular task
  (paper Sec. 6: 1-layer Transformer, ``a + b mod 31``, 50% split,
  AdamW lr 1e-3, betas (0.9, 0.98), weight decay 1). The loss is the
  standard or the fixed cross-entropy evaluated at the chosen logit
  precision. Tracked per checkpoint: train loss, test accuracy, mean
  logit margin, and the number of loss spikes after memorisation
  (the slingshot signature, paper Sec. 4).
* :func:`train_distillation` — paper Sec. 7: a student initialised
  with the teacher's weights is distilled with soft-target
  cross-entropy at temperature 4; the true gradient is zero, so any
  drift (KL growth, accuracy loss) is pure rounding error amplified
  by Adam.

Everything is seeded; rerunning with the same arguments reproduces
identical results (verified by ``tests/test_experiment.py``).
"""

from __future__ import annotations

import time

import numpy as np

from .data import TASKS, make_split
from .losses import margin as margin_fn
from .losses import soft_ce_fixed, soft_ce_standard
from .model import AdamW, Transformer


def _spike_count(losses):
    """Count distinct slingshot spike *events*: transitions into a state
    where the loss exceeds 10x the running minimum, after the run has
    memorised (loss below 0.01). A spike that persists for many steps
    counts once."""
    events = 0
    running_min = np.inf
    memorised = False
    in_spike = False
    for value in losses:
        if value < 0.01:
            memorised = True
        if memorised and running_min < 0.01:
            if not in_spike and value > 10.0 * running_min:
                events += 1
                in_spike = True
            elif in_spike and value <= 2.0 * running_min:
                in_spike = False
        running_min = min(running_min, value)
    return events


def _mean_margin(model, tokens, targets):
    logits = model.forward(tokens)
    return float(np.mean(margin_fn(logits, targets)))


def train_grokking(task="add", loss="fixed", precision="fp32", eps=1e-8,
                   steps=4000, seed=0, lr=1e-3, log_every=250,
                   weight_decay=None, d_model=64, n_heads=4, d_mlp=256,
                   split_seed=0, return_model=False):
    """One grokking run. Returns a dict with the full trajectory.

    With ``return_model=True`` the trained model is included under the
    ``"model"`` key (used to build a distillation teacher).
    """
    spec = TASKS[task]
    wd = spec["weight_decay"] if weight_decay is None else weight_decay
    tr_tok, tr_y, te_tok, te_y, vocab = make_split(task, seed=split_seed)
    model = Transformer(vocab, d_model=d_model, n_heads=n_heads,
                        d_mlp=d_mlp, seed=seed)
    opt = AdamW(model.params, lr=lr, betas=(0.9, 0.98),
                weight_decay=wd, eps=eps)
    traj = []
    t0 = time.time()
    train_losses = []
    for step in range(1, steps + 1):
        loss_val, grads = model.loss_and_grads(tr_tok, tr_y, loss=loss,
                                               precision=precision)
        opt.update(grads)
        train_losses.append(loss_val)
        if step % log_every == 0 or step == steps:
            test_acc = float((model.predict(te_tok) == te_y).mean())
            train_acc = float((model.predict(tr_tok) == tr_y).mean())
            mean_m = _mean_margin(model, tr_tok, tr_y)
            traj.append({
                "step": step,
                "train_loss": float(loss_val),
                "train_acc": train_acc,
                "test_acc": test_acc,
                "mean_margin": mean_m,
            })
    final = traj[-1]
    # Steps to grok: first checkpoint with test accuracy >= 0.99.
    grok_step = None
    for row in traj:
        if row["test_acc"] >= 0.99:
            grok_step = row["step"]
            break
    result = {
        "task": task, "loss": loss, "precision": precision, "eps": eps,
        "steps": steps, "seed": seed, "lr": lr, "weight_decay": wd,
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "final_mean_margin": final["mean_margin"],
        "grok_step": grok_step,
        "n_spikes": _spike_count(train_losses),
        "max_train_loss": float(max(train_losses)) if train_losses else None,
        "wall_seconds": round(time.time() - t0, 1),
        "trajectory": traj,
    }
    if return_model:
        result["model"] = model
    return result


def train_teacher(task="add", steps=4000, seed=0, eps=1e-8,
                  split_seed=0, **kwargs):
    """Train a teacher to (near-)100% test accuracy with the fixed loss.

    Returns ``(model, result_dict)``; the caller owns the model, so a
    distillation run can start from the teacher's exact weights.
    """
    result = train_grokking(task=task, loss="fixed", precision="fp32",
                            eps=eps, steps=steps, seed=seed,
                            split_seed=split_seed, return_model=True,
                            **kwargs)
    model = result.pop("model")
    return model, result


def _kl_div(p, q):
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(p > 0, p * (np.log(p) - np.log(np.maximum(q, 1e-300))), 0.0)
    return float(np.mean(np.sum(kl, axis=1)))


def train_distillation(teacher_model, task="add", loss="fixed",
                       precision="bf16", steps=2000, seed=0, lr=2e-3,
                       temperature=4.0, eps=1e-8, log_every=250,
                       split_seed=0):
    """Distil a student initialised at the teacher's weights (paper Sec. 7).

    The student starts as an exact copy of the teacher, so the true
    gradient is zero and the student should not move. Any KL growth or
    accuracy loss is rounding error amplified by Adam. The teacher and
    student share the task split; the teacher is only evaluated, never
    updated.
    """
    from copy import deepcopy

    tr_tok, tr_y, te_tok, te_y, vocab = make_split(task, seed=split_seed)
    student = Transformer(vocab, seed=seed)
    student.params = deepcopy(teacher_model.params)
    opt = AdamW(student.params, lr=lr, betas=(0.9, 0.98),
                weight_decay=0.0, eps=eps)
    fn = soft_ce_fixed if loss == "fixed" else soft_ce_standard
    # Teacher logits are constant across steps; compute them once.
    logits_t_tr = teacher_model.forward(tr_tok)
    logits_t_te = teacher_model.forward(te_tok)
    z_t_tr = (logits_t_tr / temperature).astype(np.float64)
    z_t_te = (logits_t_te / temperature).astype(np.float64)

    def probs(z):
        zm = z - z.max(axis=1, keepdims=True)
        e = np.exp(zm)
        return e / e.sum(axis=1, keepdims=True)

    p_t_te = probs(z_t_te)
    traj = []
    t0 = time.time()
    for step in range(1, steps + 1):
        logits_s = student.forward(tr_tok)
        z_s = (logits_s / temperature).astype(np.float64)
        loss_val, d_logits = fn(z_s, z_t_tr, dtype=precision)
        # The soft-target gradient is w.r.t. z_s / T; chain rule gives
        # a factor 1/T for the gradient w.r.t. the raw logits.
        grads = student.backward((d_logits / temperature).astype(np.float32))
        opt.update(grads)
        if step % log_every == 0 or step == steps:
            ls_te = student.forward(te_tok)
            kl = _kl_div(p_t_te, probs(ls_te / temperature))
            traj.append({
                "step": step,
                "loss": float(loss_val),
                "kl_teacher_student": kl,
                "test_acc": float((ls_te.argmax(axis=1) == te_y).mean()),
            })
    final = traj[-1]
    return {
        "task": task, "loss": loss, "precision": precision, "steps": steps,
        "seed": seed, "lr": lr, "temperature": temperature, "eps": eps,
        "final_kl": final["kl_teacher_student"],
        "final_test_acc": final["test_acc"],
        "final_loss": final["loss"],
        "wall_seconds": round(time.time() - t0, 1),
        "trajectory": traj,
    }
