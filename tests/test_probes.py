"""Tests for the exact probes: cliff locations, gradient sums, and the
cancellation residual must match the paper's Figure 1 magnitudes.
"""

import numpy as np

from softmax_cliff.probes import (cancellation_perturbed,
                                  cancellation_scan, cliff_scan,
                                  cliff_summary, equilibrium_margin,
                                  gradient_sum_scan)


def test_cliff_locations_near_paper():
    """Measured cliffs are within 1.0 of the paper's 5.5 / 16.6 / 36.7
    (small differences come from the probe's logit construction)."""
    rows = cliff_scan(margins=np.arange(0.0, 40.01, 0.25))
    summary = {(r["precision"], r["loss"]): r for r in cliff_summary(rows)}
    for precision, paper in (("bf16", 5.5), ("fp32", 16.6), ("fp64", 36.7)):
        measured = summary[(precision, "standard")]["cliff"]
        assert measured is not None
        assert abs(measured - paper) <= 1.0, (precision, measured, paper)
        assert summary[(precision, "fixed")]["cliff"] is None


def test_fixed_grad_accurate_past_cliff():
    """The fixed target gradient stays accurate past every cliff, where
    the standard one is exactly zero. Accuracy is measured against the
    exact gradient of the *representable* logits. In bf16 the tolerance
    is 10%: subtracting two bf16 logits of magnitude ~20 loses ~0.06 of
    absolute precision, and the gradient is exponential in that
    difference — a property of the dtype, not of the fix. In fp32/fp64
    the fixed gradient is accurate to 1e-4 relative."""
    from softmax_cliff.losses import dtype_of, margin as margin_fn

    rows = cliff_scan(margins=np.arange(0.0, 40.01, 0.5))
    by_key = {}
    for row in rows:
        by_key[(row["precision"], row["loss"], row["margin"])] = row
    for precision in ("bf16", "fp32", "fp64"):
        dt = dtype_of(precision)
        rtol = 0.10 if precision == "bf16" else 1e-4
        for m in (10.0, 20.0, 30.0):
            z = np.array([m, -np.log(2), -np.log(2)])
            m_repr = float(margin_fn(z.astype(dt), np.array([0]))[0])
            exact = np.exp(-m_repr)
            fixed = by_key[(precision, "fixed", m)]
            assert abs(abs(fixed["grad_t"]) - exact) < rtol * exact + 1e-12
            assert abs(fixed["grad_sum"]) < 1e-9


def test_gradient_sum_scan():
    """Standard gradient sums are positive past the cliff, fixed are 0."""
    rows = gradient_sum_scan()
    for row in rows:
        if row["loss"] == "fixed":
            assert abs(row["grad_sum"]) < 1e-9
        elif row["margin"] >= 20.0 and row["precision"] in ("bf16", "fp32"):
            assert row["grad_sum"] > 0.0


def test_cancellation_scan_magnitudes():
    """Residual medians match the paper's Fig. 1(b): ~1e-3 bf16,
    ~1e-8 fp32, ~1e-17 fp64; the fixed version is exactly zero."""
    rows = cancellation_scan(n_teachers=50, seed=7)
    by_precision = {r["precision"]: r for r in rows}
    assert 1e-4 < by_precision["bf16"]["standard_median"] < 1e-2
    assert 1e-9 < by_precision["fp32"]["standard_median"] < 1e-7
    assert by_precision["fp64"]["standard_median"] < 1e-15
    for row in rows:
        assert row["fixed_max"] == 0.0
        assert row["standard_frac_nonzero"] == 1.0


def test_cancellation_perturbed_matches_exact():
    """On the paper's perturbed example the fixed gradient matches the
    fp64 exact gradient to 1e-10; the standard one does not."""
    out = cancellation_perturbed([2.0, 1.0, 0.0], [-1e-7, 1e-7, 0.0],
                                 precision="fp32")
    assert out["fixed_max_err"] < 1e-10
    assert out["standard_max_err"] > 1e-8


def test_equilibrium_margin():
    """Paper Eq. 11: m* = log(1/eps)."""
    assert abs(equilibrium_margin(1e-8) - 18.4207) < 1e-3
    assert abs(equilibrium_margin(1e-12) - 27.6310) < 1e-3
    assert abs(equilibrium_margin(1e-16) - 36.8414) < 1e-3
