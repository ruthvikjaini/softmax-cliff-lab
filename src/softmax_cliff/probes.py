"""Exact probes of the two bugs — no training involved.

These reproduce the paper's Figure 1 numerically:

* :func:`cliff_scan` — for a grid of logit margins, the target
  gradient of the standard and fixed hard-target cross-entropy in
  bf16 / fp32 / fp64, and the first margin where the standard target
  gradient rounds to exactly zero (the precision cliff).
* :func:`gradient_sum_scan` — the sum of the loss gradient at fixed
  margins (exact value: zero; standard is positive past the cliff).
* :func:`cancellation_scan` — over random teacher logits, the residual
  gradient ``||q - p||`` of the standard soft-target cross-entropy
  when student == teacher, versus the fixed version (exactly zero).

All probes are deterministic (fixed seeds) and run in milliseconds.
"""

from __future__ import annotations

import numpy as np

from .losses import (PAPER_CLIFFS, dtype_of, exact_soft_ce_grad,
                     hard_ce_fixed, hard_ce_standard, soft_ce_fixed,
                     soft_ce_standard)


def _logits_with_margin(margin_value, n_classes=3):
    """Logits whose target margin is exactly ``margin_value``.

    Target logit ``z_t = m``; the other logits share a log-sum-exp of
    0, so ``m = z_t - logsumexp(others)`` exactly (in fp64).
    """
    others = np.full(n_classes - 1, -np.log(n_classes - 1))
    return np.concatenate([[margin_value], others])


def cliff_scan(margins=None, precisions=("bf16", "fp32", "fp64"),
               n_classes=3):
    """Scan the target gradient versus margin for both losses.

    Returns a list of rows ``{precision, loss, margin, grad_t,
    grad_sum, cliff}`` where ``cliff`` is the first margin at which
    ``|grad_t|`` is exactly zero (``None`` if it never happens on the
    grid). ``grad_t`` is the target-logit gradient of the *mean* loss
    for a single example.
    """
    if margins is None:
        margins = np.arange(0.0, 40.01, 0.25)
    rows = []
    for precision in precisions:
        for loss_name, fn in (("standard", hard_ce_standard),
                              ("fixed", hard_ce_fixed)):
            cliff = None
            for m in margins:
                z = _logits_with_margin(float(m), n_classes)
                _, g = fn(z, np.array([0]), dtype=precision)
                g_t = float(g[0, 0])
                rows.append({
                    "precision": precision,
                    "loss": loss_name,
                    "margin": float(m),
                    "grad_t": g_t,
                    "grad_sum": float(g.sum()),
                    "cliff": None,
                })
                if cliff is None and g_t == 0.0:
                    cliff = float(m)
            for row in rows:
                if row["precision"] == precision and row["loss"] == loss_name:
                    row["cliff"] = cliff
    return rows


def cliff_summary(rows):
    """Collapse a cliff scan to one row per (precision, loss)."""
    out = {}
    for row in rows:
        key = (row["precision"], row["loss"])
        out.setdefault(key, {"precision": row["precision"],
                             "loss": row["loss"],
                             "cliff": row["cliff"],
                             "paper_cliff": PAPER_CLIFFS[row["precision"]]})
    return [out[k] for k in sorted(out)]


def gradient_sum_scan(margins=(5.0, 10.0, 20.0, 30.0),
                      precisions=("bf16", "fp32", "fp64"), n_classes=3):
    """Gradient sums at fixed margins; exact value is zero."""
    rows = []
    for precision in precisions:
        for m in margins:
            z = _logits_with_margin(float(m), n_classes)
            exact = float(np.exp(-m))
            for loss_name, fn in (("standard", hard_ce_standard),
                                  ("fixed", hard_ce_fixed)):
                _, g = fn(z, np.array([0]), dtype=precision)
                rows.append({
                    "precision": precision,
                    "margin": float(m),
                    "loss": loss_name,
                    "grad_t": float(g[0, 0]),
                    "grad_sum": float(g.sum()),
                    "exact_abs_grad_t": exact,
                })
    return rows


def cancellation_scan(n_teachers=200, n_classes=8, seed=1,
                      precisions=("bf16", "fp32", "fp64")):
    """Residual gradient when student logits equal teacher logits.

    For each random teacher, compute ``max |g|`` of the standard and
    fixed soft-target cross-entropy at ``z_S == z_T``. The exact
    gradient is zero; the fixed version returns exactly zero, the
    standard version a rounding residual (paper Fig. 1(b): about
    1e-3 in bf16, 1e-8 in fp32, 1e-17 in fp64).
    """
    rng = np.random.default_rng(seed)
    teachers = rng.normal(0.0, 2.0, size=(n_teachers, n_classes))
    rows = []
    for precision in precisions:
        std_res, fixed_res = [], []
        for z_t in teachers:
            _, g_std = soft_ce_standard(z_t, z_t, dtype=precision)
            _, g_fix = soft_ce_fixed(z_t, z_t, dtype=precision)
            std_res.append(float(np.abs(g_std).max()))
            fixed_res.append(float(np.abs(g_fix).max()))
        std_res = np.array(std_res)
        rows.append({
            "precision": precision,
            "n_teachers": n_teachers,
            "standard_median": float(np.median(std_res)),
            "standard_mean": float(np.mean(std_res)),
            "standard_max": float(np.max(std_res)),
            "standard_frac_nonzero": float((std_res > 0).mean()),
            "fixed_max": float(np.max(fixed_res)),
        })
    return rows


def cancellation_perturbed(z_teacher, perturbation, precision="fp32"):
    """Compare both soft-target gradients against the fp64 exact one.

    ``perturbation`` is added to the teacher logits to form the student
    logits (the paper's example is ``[-1e-7, 1e-7, 0]`` on
    ``[2, 1, 0]``). Returns a dict with the three gradients and the
    max absolute error of each implementation versus exact.
    """
    z_t = np.asarray(z_teacher, dtype=np.float64)
    z_s = z_t + np.asarray(perturbation, dtype=np.float64)
    # Exact gradient w.r.t. the dtype-representable logits.
    dt = dtype_of(precision)
    exact = exact_soft_ce_grad(z_s.astype(dt).astype(np.float64),
                               z_t.astype(dt).astype(np.float64))[0]
    _, g_std = soft_ce_standard(z_s, z_t, dtype=precision)
    _, g_fix = soft_ce_fixed(z_s, z_t, dtype=precision)
    return {
        "precision": precision,
        "exact": [float(v) for v in exact],
        "standard": [float(v) for v in g_std[0]],
        "fixed": [float(v) for v in g_fix[0]],
        "standard_max_err": float(np.max(np.abs(g_std[0] - exact))),
        "fixed_max_err": float(np.max(np.abs(g_fix[0] - exact))),
    }


def equilibrium_margin(eps):
    """Paper Eq. 11: the margin equilibrium ``m* = log(1/eps)``."""
    return float(np.log(1.0 / eps))
