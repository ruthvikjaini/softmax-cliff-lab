"""Tests for the training experiments: determinism, output shape, and
that a short run memorises (train accuracy rises, loss falls).
"""

import numpy as np

from softmax_cliff.data import answers, make_split
from softmax_cliff.experiment import train_distillation, train_grokking


def test_split_covers_all_pairs_once():
    tr_tok, tr_y, te_tok, te_y, vocab = make_split("add", seed=0)
    assert vocab == 32
    assert len(tr_tok) + len(te_tok) == 31 * 31
    assert len(tr_tok) == 480  # 50% of 961, rounded down
    # Train and test are disjoint.
    tr_set = {tuple(r) for r in tr_tok[:, :2]}
    te_set = {tuple(r) for r in te_tok[:, :2]}
    assert not tr_set & te_set
    # Labels are (a + b) mod 31.
    np.testing.assert_array_equal(
        tr_y, answers(tr_tok[:, :2], 31, "add"))


def test_short_run_memorises_and_is_deterministic():
    kwargs = dict(task="add", loss="fixed", precision="fp32", steps=300,
                  seed=0, log_every=100)
    r1 = train_grokking(**kwargs)
    r2 = train_grokking(**kwargs)
    r1.pop("wall_seconds")
    r2.pop("wall_seconds")
    assert r1 == r2  # identical rerun apart from wall time
    assert r1["final_train_loss"] < 1.0
    assert r1["final_train_acc"] > 0.5
    assert len(r1["trajectory"]) == 3
    assert set(r1) >= {"final_test_acc", "final_mean_margin", "grok_step",
                       "n_spikes"}


def test_distillation_student_starts_at_teacher():
    """With the fixed loss, a student starting at the teacher's weights
    must stay there (KL ~ 0, accuracy unchanged) over a short run."""
    from softmax_cliff.experiment import train_teacher

    teacher, _ = train_teacher(task="add", steps=300, seed=0)
    result = train_distillation(teacher, task="add", loss="fixed",
                                precision="fp32", steps=100, seed=0,
                                log_every=50)
    assert result["final_kl"] < 1e-9
    assert result["trajectory"][0]["kl_teacher_student"] < 1e-9
