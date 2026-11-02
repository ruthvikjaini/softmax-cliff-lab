"""Tests for the standard / fixed loss functions.

Reference values are the paper's own worked examples (Barron et al.,
arXiv:2610.31642, Sec. 2–3) and fp64 exact gradients.
"""

import numpy as np
import pytest

from softmax_cliff.losses import (
    exact_hard_ce_grad,
    exact_soft_ce_grad,
    hard_ce_fixed,
    hard_ce_standard,
    log_softmax_fixed,
    log_softmax_standard,
    margin,
    soft_ce_fixed,
    soft_ce_standard,
)


def test_paper_worked_example_hard_ce():
    """Paper Sec. 2: z = [9, -9, -9], t = 0, fp32.

    Standard returns loss 0 and a zero target gradient; fixed returns
    loss 3.046e-8 and gradients summing to zero.
    """
    z = np.array([9.0, -9.0, -9.0])
    loss_std, g_std = hard_ce_standard(z, np.array([0]), dtype="fp32")
    assert loss_std == 0.0
    assert g_std[0, 0] == 0.0
    assert abs(g_std[0, 1] - 1.523e-8) < 1e-11
    loss_fix, g_fix = hard_ce_fixed(z, np.array([0]), dtype="fp32")
    assert abs(loss_fix - 3.046e-8) < 1e-11
    assert abs(g_fix[0, 0] + 3.046e-8) < 1e-11
    assert abs(g_fix[0, 1] - 1.523e-8) < 1e-11
    assert abs(g_fix.sum()) < 1e-12


def test_paper_worked_example_soft_ce():
    """Paper Sec. 3: z_T = [2, 1, 0], z_S = z_T + [-1e-7, 1e-7, 0], fp32.

    The fixed gradient matches the fp64 exact gradient; the standard
    gradient does not (wrong sign on the third entry in the paper's
    JAX numbers; here it is off by more than 30% on entry 1).
    """
    z_t = np.array([2.0, 1.0, 0.0])
    z_s = z_t + np.array([-1e-7, 1e-7, 0.0])
    # The paper's "exact" gradient is exact w.r.t. the fp32-representable
    # logits (the perturbation itself is rounded when stored in fp32).
    z_s32 = z_s.astype(np.float32).astype(np.float64)
    z_t32 = z_t.astype(np.float32).astype(np.float64)
    exact = exact_soft_ce_grad(z_s32, z_t32)[0]
    assert abs(exact[0] + 4.596e-8) < 1e-11
    assert abs(exact[1] - 4.144e-8) < 1e-11
    _, g_fix = soft_ce_fixed(z_s, z_t, dtype="fp32")
    np.testing.assert_allclose(g_fix[0], exact, atol=1e-10)
    _, g_std = soft_ce_standard(z_s, z_t, dtype="fp32")
    assert np.max(np.abs(g_std[0] - exact)) > 1e-8


def test_fixed_hard_ce_matches_exact_fp64():
    """Fixed CE gradient matches the fp64 exact gradient at all margins."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = rng.normal(0, 5, size=(4, 7))
        t = rng.integers(0, 7, size=4)
        _, g_fix = hard_ce_fixed(z, t, dtype="fp64")
        np.testing.assert_allclose(g_fix, exact_hard_ce_grad(z, t),
                                   rtol=1e-9, atol=1e-12)


def test_standard_hard_ce_matches_exact_below_cliff():
    """Standard CE is exact while the margin is below the fp32 cliff."""
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.0, size=(8, 5))
    t = rng.integers(0, 5, size=8)
    _, g_std = hard_ce_standard(z, t, dtype="fp32")
    np.testing.assert_allclose(g_std.astype(np.float64),
                               exact_hard_ce_grad(z, t),
                               rtol=1e-5, atol=1e-7)


def test_fixed_grad_sums_to_zero_past_cliff():
    """Past the cliff the fixed gradient still sums to zero exactly."""
    for precision in ("bf16", "fp32", "fp64"):
        for m in (10.0, 20.0, 30.0):
            z = np.array([m, -np.log(2), -np.log(2)])
            _, g = hard_ce_fixed(z, np.array([0]), dtype=precision)
            assert abs(float(g.sum())) < 1e-12, (precision, m, g)


def test_standard_grad_sum_positive_past_cliff():
    """Past the cliff the standard gradient sum is the missing target
    gradient, ``exp(-m) / (1 + exp(-m))`` (paper Eq. 4)."""
    for m in (20.0, 30.0):
        z = np.array([m, -np.log(2), -np.log(2)])
        _, g = hard_ce_standard(z, np.array([0]), dtype="fp32")
        expected = np.exp(-m) / (1 + np.exp(-m))
        assert abs(float(g.sum()) - expected) < 1e-12


def test_fixed_soft_ce_zero_at_equal_logits():
    """Fixed soft-target CE gradient is exactly zero when z_S == z_T,
    in every precision (paper Sec. 3)."""
    rng = np.random.default_rng(2)
    for precision in ("bf16", "fp32", "fp64"):
        for _ in range(10):
            z = rng.normal(0, 2, size=(3, 6))
            _, g = soft_ce_fixed(z, z, dtype=precision)
            assert np.all(g == 0.0), precision


def test_standard_soft_ce_nonzero_at_equal_logits():
    """Standard soft-target CE leaves a non-zero residual gradient at
    z_S == z_T (the cancellation bug), in fp32 and fp64."""
    rng = np.random.default_rng(3)
    z = rng.normal(0, 2, size=(3, 6))
    for precision in ("fp32", "fp64"):
        _, g = soft_ce_standard(z, z, dtype=precision)
        assert np.abs(g).max() > 0.0, precision


def test_fixed_log_softmax_matches_standard_below_cliff():
    """Below the cliff the fixed and standard log-softmax agree."""
    rng = np.random.default_rng(4)
    z = rng.normal(0, 1.0, size=(8, 5))
    a = log_softmax_fixed(z, dtype="fp32").astype(np.float64)
    b = log_softmax_standard(z, dtype="fp32").astype(np.float64)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_margin_definition():
    """Margin is z_t minus logsumexp of the other logits."""
    z = np.array([[3.0, 1.0, 0.0]])
    m = margin(z, np.array([0]))
    expected = 3.0 - np.log(np.exp(1.0) + np.exp(0.0))
    assert abs(m[0] - expected) < 1e-12
