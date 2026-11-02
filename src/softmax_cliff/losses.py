"""Standard and fixed softmax / log-softmax / cross-entropy.

Implements, from the equations of Barron, Gnanapragasam and Tschannen,
"A Fix for Softmax and Cross-Entropy" (arXiv:2610.31642, v3, 29 Oct 2026):

* Standard implementations (paper Eq. 1): max-shift log-softmax,
  ``logsoftmax(z)_k = (z_k - z_max) - log sum_j exp(z_j - z_max)``.
  These carry the two bugs below.

* Bug 1 — Softmax Collapse (paper Sec. 2): once the logit margin
  ``m = z_t - logsumexp_{j != t} z_j`` exceeds the precision cliff
  ``c_d`` (about 5.5 in bf16, 16.6 in fp32, 36.7 in fp64), the
  denominator ``1 + exp(-m)`` rounds to exactly 1, so the target
  gradient ``q_t - 1`` rounds to zero while the non-target gradients
  stay positive: the gradients no longer sum to zero.

* Bug 2 — Softmax Cancellation (paper Sec. 3): in soft-target
  distillation the gradient ``q - p`` is computed by subtracting two
  nearly equal probabilities evaluated through different op orders,
  leaving a non-zero residual gradient even when student == teacher.

* Fixed log-softmax / hard-target cross-entropy (paper Eq. 5–7):
  the argmax (or target) slot is set to exactly 0 and excluded from
  the sum of exponentials; the loss is ``softplus(-m)`` and the
  target gradient is the negated sum of the non-target gradients,
  so it stays accurate down to exponent underflow and the gradient
  sums to zero by construction.

* Fixed soft-target cross-entropy (paper Eq. 9): ``q - p`` is computed
  in log space from ``delta = z_S - z_T`` via ``expm1``/``log1p``,
  centred under ``p``; the gradient is exactly zero when
  ``z_S == z_T``.

All functions are parameterised by a floating-point dtype
(``"bf16"``, ``"fp32"``, ``"fp64"``) and perform their arithmetic in
that dtype, so the precision cliff is reproduced with real low-precision
arithmetic rather than simulated. ``bf16`` is the bfloat16 type from
the ``ml_dtypes`` package.

Every function returning a gradient returns the *analytic* gradient of
the loss it computes, as a pair ``(loss, grad)``; the model in
:mod:`softmax_cliff.model` calls these and backpropagates the gradient
through the network. Finite-difference checks in ``tests/`` verify the
analytic gradients against fp64 references.
"""

from __future__ import annotations

import numpy as np

try:
    import ml_dtypes
except ImportError:  # pragma: no cover - ml_dtypes is a hard dependency
    ml_dtypes = None  # type: ignore[assignment]

DTYPES = {
    "bf16": None if ml_dtypes is None else ml_dtypes.bfloat16,
    "fp32": np.float32,
    "fp64": np.float64,
}

# Precision cliffs from the paper (Figure 1): the margin above which the
# standard target gradient rounds to zero.
PAPER_CLIFFS = {"bf16": 5.5, "fp32": 16.6, "fp64": 36.7}


def dtype_of(name: str):
    """Resolve a precision name to a NumPy dtype."""
    if name not in DTYPES or DTYPES[name] is None:
        raise ValueError(f"unknown or unavailable precision {name!r}; "
                         f"expected one of {sorted(DTYPES)}")
    return DTYPES[name]


def _softplus(x):
    """Numerically stable ``log(1 + exp(x))`` in the input's dtype."""
    # For x > 0: x + log1p(exp(-x)); else log1p(exp(x)). No overflow.
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))


def _lse_others(z, t):
    """``logsumexp`` of all entries except index ``t``, per row.

    ``z`` is a ``(N, K)`` array, ``t`` a ``(N,)`` index array.
    Computed with a max-shift so it is stable for large margins.
    Returns a ``(N,)`` array in ``z``'s dtype.
    """
    n, k = z.shape
    rows = np.arange(n)
    z_others = z.copy()
    z_others[rows, t] = -np.inf
    mx = z_others.max(axis=1, keepdims=True)
    # If every other entry is -inf (degenerate K == 1), the sum is 0 and
    # log(0) is -inf; that propagates correctly as margin +inf.
    with np.errstate(divide="ignore", invalid="ignore"):
        lse = mx[:, 0] + np.log(np.sum(np.exp(z_others - mx), axis=1))
    return lse


def log_softmax_standard(z, dtype="fp32"):
    """Standard max-shift log-softmax (paper Eq. 1)."""
    dt = dtype_of(dtype)
    z = np.asarray(z, dtype=dt)
    zm = z - z.max(axis=-1, keepdims=True)
    return zm - np.log(np.sum(np.exp(zm), axis=-1, keepdims=True))


def log_softmax_fixed(z, dtype="fp32"):
    """Fixed log-softmax (paper Eq. 5).

    The argmax slot is set to exactly 0 and excluded from the sum of
    exponentials; the normaliser is ``log1p(sum of the other exps)``.
    """
    dt = dtype_of(dtype)
    z = np.asarray(z, dtype=dt)
    i_max = np.argmax(z, axis=-1)
    rows = np.arange(z.shape[0]) if z.ndim == 2 else None
    if z.ndim == 1:
        z = z[None, :]
        i_max = np.atleast_1d(i_max)
        rows = np.arange(1)
        single = True
    else:
        single = False
    z_max = z[rows, i_max][:, None]
    is_max = np.zeros_like(z, dtype=bool)
    is_max[rows, i_max] = True
    shifted = np.where(is_max, dt(0.0), z - z_max)
    exp_others = np.where(is_max, dt(0.0), np.exp(shifted))
    s = np.sum(exp_others, axis=-1, keepdims=True)
    out = shifted - np.log1p(s)
    return out[0] if single else out


def softmax_standard(z, dtype="fp32"):
    """Standard softmax, as evaluated in a forward pass (paper Eq. 1)."""
    dt = dtype_of(dtype)
    z = np.asarray(z, dtype=dt)
    zm = z - z.max(axis=-1, keepdims=True)
    e = np.exp(zm)
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_fixed(z, dtype="fp32"):
    """Fixed softmax: ``exp`` of the fixed log-softmax (paper Eq. 5)."""
    return np.exp(log_softmax_fixed(z, dtype=dtype))


def hard_ce_standard(logits, targets, dtype="fp32"):
    """Standard hard-target cross-entropy (paper Eq. 1).

    Returns ``(loss, grad)`` where ``loss`` is the mean loss (a float)
    and ``grad`` is the ``(N, K)`` gradient of the mean loss w.r.t. the
    logits, in ``dtype``. The gradient is ``q - one_hot(target)`` with
    ``q`` from the standard softmax; it is the gradient autodiff
    produces for this forward pass, including the collapse bug.
    """
    dt = dtype_of(dtype)
    z = np.asarray(logits, dtype=dt)
    if z.ndim == 1:
        z = z[None, :]
    targets = np.asarray(targets)
    n = z.shape[0]
    rows = np.arange(n)
    q = softmax_standard(z, dtype=dtype)
    with np.errstate(divide="ignore"):
        loss = float(np.mean(-np.log(q[rows, targets].astype(np.float64))))
    g = q.copy()
    g[rows, targets] = g[rows, targets] - dt(1.0)
    return loss, (g / dt(n)).astype(dt)


def hard_ce_fixed(logits, targets, dtype="fp32"):
    """Fixed hard-target cross-entropy (paper Eq. 6–7).

    Loss ``L = softplus(-m)`` with ``m`` the logit margin; gradients
    ``g_j = exp(z_j - z_t - L)`` for ``j != t`` and ``g_t`` the negated
    sum of the others, so the gradient sums to zero by construction
    and stays accurate past the precision cliff.
    """
    dt = dtype_of(dtype)
    z = np.asarray(logits, dtype=dt)
    if z.ndim == 1:
        z = z[None, :]
    targets = np.asarray(targets)
    n = z.shape[0]
    rows = np.arange(n)
    zt = z[rows, targets][:, None]
    diffs = z - zt
    m = -_lse_others(diffs, targets)
    loss_per = _softplus(-m)
    loss = float(np.mean(loss_per.astype(np.float64)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        g = np.exp(diffs - loss_per[:, None])
    g[rows, targets] = dt(0.0)
    g[rows, targets] = -np.sum(g, axis=1)
    return loss, (g / dt(n)).astype(dt)


def soft_ce_standard(z_student, z_teacher, dtype="fp32"):
    """Standard soft-target cross-entropy (paper Eq. 8).

    The teacher distribution ``p`` comes from the forward-pass softmax
    while the student distribution ``q`` comes from ``exp`` of the
    backward-path log-softmax, mirroring the op-order mismatch in
    framework implementations that produces the cancellation bug.
    Returns ``(loss, grad)`` w.r.t. the student logits.
    """
    dt = dtype_of(dtype)
    z_s = np.asarray(z_student, dtype=dt)
    z_t = np.asarray(z_teacher, dtype=dt)
    if z_s.ndim == 1:
        z_s, z_t = z_s[None, :], z_t[None, :]
    p = softmax_standard(z_t, dtype=dtype)
    log_q = log_softmax_standard(z_s, dtype=dtype)
    q = np.exp(log_q)
    with np.errstate(divide="ignore"):
        loss = float(np.mean(-np.sum(p * log_q, axis=1).astype(np.float64)))
    g = q - p
    return loss, (g / dt(z_s.shape[0])).astype(dt)


def soft_ce_fixed(z_student, z_teacher, dtype="fp32"):
    """Fixed soft-target cross-entropy (paper Eq. 9).

    ``q - p`` is computed in log space from ``delta = z_S - z_T`` via
    ``expm1``/``log1p``, centred under ``p``; the gradient is exactly
    zero when ``z_S == z_T`` and matches the fp64 exact gradient to
    rounding otherwise. The forward loss is ``-sum p * log_q`` with the
    fixed log-softmax.
    """
    dt = dtype_of(dtype)
    z_s = np.asarray(z_student, dtype=dt)
    z_t = np.asarray(z_teacher, dtype=dt)
    if z_s.ndim == 1:
        z_s, z_t = z_s[None, :], z_t[None, :]
    log_q = log_softmax_fixed(z_s, dtype=dtype)
    q = np.exp(log_q)
    p = softmax_fixed(z_t, dtype=dtype)
    with np.errstate(divide="ignore"):
        loss = float(np.mean(-np.sum(p * log_q, axis=1).astype(np.float64)))
    delta = z_s - z_t
    d = delta - np.sum(p * delta, axis=-1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        s = d - np.log1p(np.sum(p * np.expm1(d), axis=-1, keepdims=True))
        u = np.where(np.abs(s) < 1, p * np.expm1(s), q - p)
    g = u - q * np.sum(u, axis=-1, keepdims=True)
    return loss, (g / dt(z_s.shape[0])).astype(dt)


def margin(logits, targets):
    """Logit margin ``m = z_t - logsumexp_{j != t} z_j`` (fp64)."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    targets = np.asarray(targets)
    zt = z[np.arange(z.shape[0]), targets]
    return zt - _lse_others(z, targets).astype(np.float64)


def exact_hard_ce_grad(logits, targets):
    """fp64 reference gradient of hard-target CE (no rounding bugs)."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    targets = np.asarray(targets)
    n = z.shape[0]
    rows = np.arange(n)
    zm = z - z.max(axis=1, keepdims=True)
    e = np.exp(zm)
    q = e / e.sum(axis=1, keepdims=True)
    g = q.copy()
    g[rows, targets] -= 1.0
    return g / n


def exact_soft_ce_grad(z_student, z_teacher):
    """fp64 reference gradient of soft-target CE w.r.t. student logits."""
    z_s = np.asarray(z_student, dtype=np.float64)
    z_t = np.asarray(z_teacher, dtype=np.float64)
    if z_s.ndim == 1:
        z_s, z_t = z_s[None, :], z_t[None, :]
    def sm(z):
        zm = z - z.max(axis=1, keepdims=True)
        e = np.exp(zm)
        return e / e.sum(axis=1, keepdims=True)
    return (sm(z_s) - sm(z_t)) / z_s.shape[0]
