"""Modular-arithmetic tasks and train/test splits.

Tasks (paper Sec. 6 and Fig. 6, following Varma et al. 2023, Nanda
et al. 2023, Doshi et al. 2024):

* ``add``: ``(a + b) mod P`` — the paper's Table 1 task (P = 31).
* ``add113``: ``(a + b) mod 113`` — paper Fig. 6(a).
* ``sumsq``: ``(a^2 + b^2) mod P`` — paper Fig. 6(c) (P = 97).
* ``cube``: ``(a^3 + b) mod P`` — paper Fig. 6(d) (P = 97).

Inputs are token sequences ``[a, b, eq]`` where ``eq`` is a dedicated
separator token id ``P`` (so the vocabulary size is ``P + 1``).
"""

from __future__ import annotations

import numpy as np

TASKS = {
    "add": {"modulus": 31, "op": "add", "weight_decay": 1.0},
    "add113": {"modulus": 113, "op": "add", "weight_decay": 1.0},
    "sumsq": {"modulus": 97, "op": "sumsq", "weight_decay": 0.2},
    "cube": {"modulus": 97, "op": "cube", "weight_decay": 0.2},
}


def all_pairs(modulus):
    """All ``(a, b)`` pairs as a ``(P*P, 2)`` int array."""
    a = np.repeat(np.arange(modulus), modulus)
    b = np.tile(np.arange(modulus), modulus)
    return np.stack([a, b], axis=1)


def answers(pairs, modulus, op):
    a, b = pairs[:, 0], pairs[:, 1]
    if op == "add":
        return (a + b) % modulus
    if op == "sumsq":
        return (a * a + b * b) % modulus
    if op == "cube":
        return (a * a * a + b) % modulus
    raise ValueError(f"unknown op {op!r}")


def make_split(task="add", train_frac=0.5, seed=0):
    """Return ``(train_tokens, train_y, test_tokens, test_y, vocab)``.

    The paper uses a 50% training split of all pairs, shuffled with a
    fixed seed. Tokens are ``[a, b, P]`` with ``P`` the separator id.
    """
    spec = TASKS[task]
    p = spec["modulus"]
    pairs = all_pairs(p)
    y = answers(pairs, p, spec["op"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_train = int(len(pairs) * train_frac)
    tr, te = idx[:n_train], idx[n_train:]
    sep = np.full((len(pairs), 1), p, dtype=np.int64)
    tokens = np.concatenate([pairs, sep], axis=1)
    return tokens[tr], y[tr], tokens[te], y[te], p + 1
