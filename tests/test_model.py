"""Tests for the Transformer: shapes, causality, manual-backprop
correctness (finite differences), and AdamW behaviour.
"""

import numpy as np
import pytest

from softmax_cliff.model import AdamW, Transformer


def _tiny_model():
    model = Transformer(vocab_size=5, d_model=8, n_heads=2, d_mlp=16,
                        seq_len=3, seed=0)
    model.params = {k: v.astype(np.float64) for k, v in model.params.items()}
    return model


def test_forward_shapes():
    model = Transformer(vocab_size=32, seed=1)
    tokens = np.array([[0, 1, 31], [5, 6, 31], [30, 29, 31]])
    logits = model.forward(tokens)
    assert logits.shape == (3, 32)
    assert np.all(np.isfinite(logits))


def test_causal_attention():
    """Changing a later token must not change earlier positions' hidden
    states, so the readout (last position) may change but the first
    position's representation must not."""
    model = Transformer(vocab_size=8, d_model=8, n_heads=2, d_mlp=16,
                        seq_len=3, seed=2)
    base = np.array([[1, 2, 7]])
    changed = np.array([[1, 5, 7]])
    model.forward(base)
    x2_base = model.cache["x2"].copy()
    model.forward(changed)
    x2_changed = model.cache["x2"]
    # Position 0 sees only token 0 in both runs.
    np.testing.assert_allclose(x2_base[:, 0, :], x2_changed[:, 0, :],
                               rtol=0, atol=0)
    # Position 1 sees tokens 0..1, which differ, so it must change.
    assert not np.allclose(x2_base[:, 1, :], x2_changed[:, 1, :])


def test_backward_matches_finite_differences():
    """Hand-written backprop vs central finite differences, both losses."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 5, size=(4, 3))
    targets = rng.integers(0, 5, size=4)
    for loss in ("standard", "fixed"):
        model = _tiny_model()
        _, grads = model.loss_and_grads(tokens, targets, loss=loss,
                                        precision="fp64")
        for key, p in model.params.items():
            flat = p.ravel()
            gflat = grads[key].ravel()
            idxs = rng.choice(len(flat), size=min(4, len(flat)),
                              replace=False)
            for i in idxs:
                old = flat[i]
                eps = 1e-5
                flat[i] = old + eps
                lp, _ = model.loss_and_grads(tokens, targets, loss=loss,
                                             precision="fp64")
                flat[i] = old - eps
                lm, _ = model.loss_and_grads(tokens, targets, loss=loss,
                                             precision="fp64")
                flat[i] = old
                fd = (lp - lm) / (2 * eps)
                assert abs(fd - gflat[i]) < 1e-4, (loss, key, i, fd, gflat[i])


def test_adamw_step_reduces_loss():
    """A few AdamW steps on a tiny batch must reduce the training loss."""
    rng = np.random.default_rng(5)
    model = Transformer(vocab_size=6, d_model=8, n_heads=2, d_mlp=16,
                        seq_len=3, seed=3)
    tokens = rng.integers(0, 6, size=(8, 3))
    targets = rng.integers(0, 6, size=8)
    opt = AdamW(model.params, lr=1e-2, weight_decay=0.0, eps=1e-8)
    loss0, _ = model.loss_and_grads(tokens, targets, loss="fixed",
                                    precision="fp32")
    for _ in range(30):
        _, grads = model.loss_and_grads(tokens, targets, loss="fixed",
                                        precision="fp32")
        opt.update(grads)
    loss1, _ = model.loss_and_grads(tokens, targets, loss="fixed",
                                    precision="fp32")
    assert loss1 < loss0


def test_loss_functions_agree_at_init():
    """At initialisation (small margins) standard and fixed losses agree."""
    rng = np.random.default_rng(6)
    model = Transformer(vocab_size=8, d_model=8, n_heads=2, d_mlp=16,
                        seq_len=3, seed=4)
    tokens = rng.integers(0, 8, size=(8, 3))
    targets = rng.integers(0, 8, size=8)
    ls, _ = model.loss_and_grads(tokens, targets, loss="standard",
                                 precision="fp32")
    lf, _ = model.loss_and_grads(tokens, targets, loss="fixed",
                                 precision="fp32")
    assert abs(ls - lf) < 1e-4
