"""A small 1-layer Transformer for grokking, with manual backprop.

Architecture (following Barron et al. Sec. 6, which follows Varma et
al. 2023): 1 layer, 4 attention heads, width 64, ReLU MLP, no
LayerNorm, trained full-batch with AdamW on modular arithmetic.
The paper does not state the MLP width or the tokenisation, so those
are documented choices here: input sequence ``[a, b, =]`` (3 tokens),
readout from the final position, MLP hidden width 256, learned token
and positional embeddings.

All parameters are fp32. The loss is computed at a chosen logit
precision (bf16 / fp32 / fp64) by the functions in
:mod:`softmax_cliff.losses`; the loss gradient is cast back to fp32
for the backward pass, mirroring the paper's "logit precision at the
loss" setup. Attention softmax inside the model stays in fp32: the
paper's fix concerns the *loss* softmax, not attention.

Backprop is implemented by hand and verified against finite
differences in ``tests/test_model.py``.
"""

from __future__ import annotations

import numpy as np

from .losses import dtype_of, hard_ce_fixed, hard_ce_standard


def _xavier(rng, fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


class Transformer:
    """1-layer causal Transformer for modular-arithmetic grokking."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_mlp=256,
                 seq_len=3, seed=0):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp
        self.seq_len = seq_len
        rng = np.random.default_rng(seed)
        emb_scale = 1.0 / np.sqrt(d_model)
        self.params = {
            "tok_emb": (rng.normal(0, emb_scale, (vocab_size, d_model))
                        .astype(np.float32)),
            "pos_emb": (rng.normal(0, emb_scale, (seq_len, d_model))
                        .astype(np.float32)),
            "W_q": _xavier(rng, d_model, d_model),
            "W_k": _xavier(rng, d_model, d_model),
            "W_v": _xavier(rng, d_model, d_model),
            "W_o": _xavier(rng, d_model, d_model),
            "W_1": _xavier(rng, d_model, d_mlp),
            "b_1": np.zeros(d_mlp, dtype=np.float32),
            "W_2": _xavier(rng, d_mlp, d_model),
            "b_2": np.zeros(d_model, dtype=np.float32),
            "W_out": _xavier(rng, d_model, vocab_size),
            "b_out": np.zeros(vocab_size, dtype=np.float32),
        }
        self.cache = None

    # -- forward ------------------------------------------------------
    def forward(self, tokens):
        """Return logits ``(N, vocab)`` from token ids ``(N, seq_len)``."""
        p = self.params
        n = tokens.shape[0]
        x = p["tok_emb"][tokens] + p["pos_emb"][None, :, :]  # (N, T, D)
        q = x @ p["W_q"]
        k = x @ p["W_k"]
        v = x @ p["W_v"]
        # Split heads: (N, T, H, Dh) -> (N, H, T, Dh)
        qh = q.reshape(n, self.seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        kh = k.reshape(n, self.seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        vh = v.reshape(n, self.seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        scale = 1.0 / np.sqrt(self.d_head)
        scores = (qh @ kh.transpose(0, 1, 3, 2)) * scale  # (N, H, T, T)
        mask = np.triu(np.ones((self.seq_len, self.seq_len), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
        s_max = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - s_max)
        attn = e / np.sum(e, axis=-1, keepdims=True)  # (N, H, T, T)
        ctx = attn @ vh  # (N, H, T, Dh)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(n, self.seq_len, self.d_model)
        attn_out = ctx @ p["W_o"]
        x1 = x + attn_out
        h_pre = x1 @ p["W_1"] + p["b_1"]
        h = np.maximum(h_pre, 0)
        mlp_out = h @ p["W_2"] + p["b_2"]
        x2 = x1 + mlp_out
        logits = x2[:, -1, :] @ p["W_out"] + p["b_out"]
        self.cache = {
            "tokens": tokens, "x": x, "q": q, "k": k, "v": v,
            "qh": qh, "kh": kh, "vh": vh, "scores": scores,
            "attn": attn, "ctx": ctx, "attn_out": attn_out,
            "x1": x1, "h_pre": h_pre, "h": h, "mlp_out": mlp_out,
            "x2": x2,
        }
        return logits

    # -- backward -----------------------------------------------------
    def backward(self, d_logits):
        """Gradient of the mean loss w.r.t. every parameter.

        ``d_logits`` is the ``(N, vocab)`` gradient of the loss w.r.t.
        the output logits (fp32). Returns a dict of gradients with the
        same keys/shapes as ``self.params``.
        """
        p, c = self.params, self.cache
        n = c["tokens"].shape[0]
        g = {}
        # Output head (readout from final position only).
        g["W_out"] = c["x2"][:, -1, :].T @ d_logits
        g["b_out"] = d_logits.sum(axis=0)
        dx2 = np.zeros_like(c["x2"])
        dx2[:, -1, :] = d_logits @ p["W_out"].T
        # MLP block.
        d_mlp_out = dx2
        g["W_2"] = c["h"].reshape(-1, self.d_mlp).T @ d_mlp_out.reshape(-1, self.d_model)
        g["b_2"] = d_mlp_out.sum(axis=(0, 1))
        dh = d_mlp_out @ p["W_2"].T
        dh = dh * (c["h_pre"] > 0)
        g["W_1"] = c["x1"].reshape(-1, self.d_model).T @ dh.reshape(-1, self.d_mlp)
        g["b_1"] = dh.sum(axis=(0, 1))
        dx1 = dx2 + dh @ p["W_1"].T
        # Attention block.
        d_attn_out = dx1
        g["W_o"] = c["ctx"].reshape(-1, self.d_model).T @ d_attn_out.reshape(-1, self.d_model)
        d_ctx = d_attn_out @ p["W_o"].T  # (N, T, D)
        d_ctx_h = d_ctx.reshape(n, self.seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        d_attn = d_ctx_h @ c["vh"].transpose(0, 1, 3, 2)
        d_vh = c["attn"].transpose(0, 1, 3, 2) @ d_ctx_h
        # Backprop through softmax over the last axis of scores.
        s = np.sum(d_attn * c["attn"], axis=-1, keepdims=True)
        d_scores = c["attn"] * (d_attn - s)
        mask = np.triu(np.ones((self.seq_len, self.seq_len), dtype=bool), k=1)
        d_scores = np.where(mask, 0.0, d_scores)
        scale = 1.0 / np.sqrt(self.d_head)
        d_qh = (d_scores @ c["kh"]) * scale
        d_kh = (d_scores.transpose(0, 1, 3, 2) @ c["qh"]) * scale
        d_q = d_qh.transpose(0, 2, 1, 3).reshape(n, self.seq_len, self.d_model)
        d_k = d_kh.transpose(0, 2, 1, 3).reshape(n, self.seq_len, self.d_model)
        d_v = d_vh.transpose(0, 2, 1, 3).reshape(n, self.seq_len, self.d_model)
        x_flat = c["x"].reshape(-1, self.d_model)
        g["W_q"] = x_flat.T @ d_q.reshape(-1, self.d_model)
        g["W_k"] = x_flat.T @ d_k.reshape(-1, self.d_model)
        g["W_v"] = x_flat.T @ d_v.reshape(-1, self.d_model)
        dx = (d_q @ p["W_q"].T + d_k @ p["W_k"].T + d_v @ p["W_v"].T
              + d_attn_out)
        # Embedding lookup: scatter-add over token and position ids.
        # One-hot matmul instead of np.add.at (which is ~20 ms/step).
        flat_tokens = c["tokens"].ravel()
        one_hot = np.zeros((flat_tokens.shape[0], self.vocab_size),
                           dtype=np.float32)
        one_hot[np.arange(flat_tokens.shape[0]), flat_tokens] = 1.0
        g["tok_emb"] = one_hot.T @ dx.reshape(-1, self.d_model)
        g["pos_emb"] = dx.sum(axis=0)
        return g

    # -- loss + step helpers ------------------------------------------
    def loss_and_grads(self, tokens, targets, loss="fixed", precision="fp32"):
        """Mean loss and parameter gradients for one full batch."""
        logits = self.forward(tokens)
        dt = dtype_of(precision)
        fn = hard_ce_fixed if loss == "fixed" else hard_ce_standard
        loss_val, d_logits = fn(logits, targets, dtype=precision)
        grads = self.backward(d_logits.astype(np.float32))
        return loss_val, grads

    def predict(self, tokens):
        return self.forward(tokens).argmax(axis=1)


class AdamW:
    """AdamW with decoupled weight decay (paper Sec. 4/6)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0,
                 eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.step = 0

    def update(self, grads):
        self.step += 1
        b1, b2 = self.beta1, self.beta2
        for k, p in self.params.items():
            g = grads[k].astype(np.float32)
            if self.weight_decay != 0.0:
                p -= self.lr * self.weight_decay * p
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.step)
            v_hat = self.v[k] / (1 - b2 ** self.step)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
