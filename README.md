# softmax-cliff-lab

Every deep learning framework ships the same two numerical bugs in
softmax, log-softmax and cross-entropy — and they bite exactly when a
model gets confident. This repo reproduces and extends Barron,
Gnanapragasam and Tschannen (Google DeepMind), **"A Fix for Softmax
and Cross-Entropy"** (arXiv:2610.31642, v3, 29 Oct 2026), in NumPy on
CPU: the two bugs, the two-line fixes, and what they do to grokking
and distillation.

## The two bugs

**Softmax Collapse.** Standard implementations shift logits by their
maximum before exponentiating. Once the logit margin
`m = z_t - logsumexp_{j != t} z_j` passes a *precision cliff*
(about 5.5 in bf16, 16.6 in fp32, 36.7 in fp64), the denominator
`1 + exp(-m)` rounds to exactly 1, so the target logit gradient
`q_t - 1` rounds to zero — while the non-target gradients stay
positive. The gradients no longer sum to zero, and under Adam this
produces the "slingshot" loss spikes that stall grokking.

**Softmax Cancellation.** In soft-target distillation the gradient is
`q - p`, computed by subtracting two nearly equal probabilities that
were evaluated through different sequences of operations. As the
student approaches the teacher the true gradient shrinks to zero, but
the computed one plateaus at a rounding floor (~1e-3 in bf16) — and
Adam amplifies even that into real parameter drift.

## The fix

Never add 1 to `exp(-m)`. Set the argmax/target slot to exactly 0,
exclude it from the sum of exponentials, evaluate the loss as
`softplus(-m)` via `log1p`, and take the target gradient as the
negated sum of the non-target gradients — so it sums to zero by
construction and stays accurate down to exponent underflow. For soft
targets, compute `q - p` in log space from `delta = z_S - z_T` with
`expm1`/`log1p`, centred under `p`. The gradient is then exactly zero
when student == teacher. See `src/softmax_cliff/losses.py` (paper
Eq. 5–7, 9) and `docs/RESEARCH.md`.

## What this repo does

- **Exact probes** (`src/softmax_cliff/probes.py`): the precision
  cliff, gradient sums, and cancellation residuals in real bf16 /
  fp32 / fp64 arithmetic — no training, deterministic, milliseconds.
- **Grokking** (`src/softmax_cliff/experiment.py`): a 1-layer,
  4-head, width-64 ReLU Transformer (no LayerNorm) trained full-batch
  with AdamW on `a + b mod 31`, standard vs fixed cross-entropy, at
  bf16 and fp32 logit precision — the paper's Table 1 setup, reduced
  to what a CPU can do in one night (see `docs/RESEARCH.md`).
- **Distillation**: a student initialised at the teacher's exact
  weights, distilled at temperature 4 with the standard vs fixed
  soft-target loss — the true gradient is zero, so any drift is pure
  rounding error amplified by Adam (paper Sec. 7).

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/demo.py            # probes + a short grokking run
python -m pytest tests/ -q        # 25 tests
python scripts/run_experiment.py  # full grid -> results/
python scripts/crossfoot.py       # README figures vs results/
```

Requires Python 3.10+, NumPy, and `ml_dtypes` (for real bfloat16
arithmetic). CPU-only; no GPU, no model downloads, no API keys.

## Measured results

All numbers below are measured by this repo on CPU (NumPy +
`ml_dtypes`), cross-footed against `results/measured.json` by
`scripts/crossfoot.py`. Paper numbers are quoted in the prose for
comparison and are not this repo's measurements.

### Precision cliff (paper Fig. 1(a))

The first margin at which the target gradient of the standard loss
rounds to exactly zero. The paper reports cliffs of 5.5 (bf16), 16.6
(fp32) and 36.7 (fp64) from its own logit construction; the probe
here uses a 3-class construction with an exact margin, so the values
differ slightly. The fixed loss never zeroes its target gradient on
the whole 0–40 margin grid, in any precision.

| precision | standard cliff | fixed cliff |
|---|---|---|
| bf16 | 5.0 | None |
| fp32 | 16.0 | None |
| fp64 | 36.25 | None |

### Gradient sums at margin 20 (exact value: 0)

| precision | loss | grad sum | target grad | exact \|grad\| |
|---|---|---|---|---|
| bf16 | standard | 1.950e-09 | 0.000e+00 | 2.061e-09 |
| bf16 | fixed | 0.000e+00 | -1.950e-09 | 2.061e-09 |
| fp32 | standard | 2.061e-09 | 0.000e+00 | 2.061e-09 |
| fp32 | fixed | 0.000e+00 | -2.061e-09 | 2.061e-09 |
| fp64 | standard | 1.481e-16 | -2.061e-09 | 2.061e-09 |
| fp64 | fixed | 0.000e+00 | -2.061e-09 | 2.061e-09 |

Past the cliff the standard gradient sum is exactly the missing
target gradient, `exp(-m) / (1 + exp(-m))` (paper Eq. 4); the fixed
gradient sums to zero. In bf16 the fixed target gradient is
1.950e-09 rather than 2.061e-09 — a 5% relative error that comes from
bf16 rounding of the logits themselves (subtracting two bf16 numbers
of magnitude ~20 loses ~0.06 absolute precision), not from the fix.

### Softmax cancellation (paper Fig. 1(b))

Residual gradient `max |q - p|` when student logits equal teacher
logits, over 200 random teachers (8 classes). The paper reports
about 1e-3 in bf16, 1e-8 in fp32 and 1e-17 in fp64.

| precision | standard median | standard max | fixed max |
|---|---|---|---|
| bf16 | 9.77e-04 | 3.91e-03 | 0.0e+00 |
| fp32 | 2.98e-08 | 1.19e-07 | 0.0e+00 |
| fp64 | 2.78e-17 | 1.11e-16 | 0.0e+00 |

On the paper's perturbed example (`z_T = [2, 1, 0]`,
`z_S = z_T + [-1e-7, 1e-7, 0]`, fp32) the fixed gradient matches the
fp64 exact gradient to 1.03e-15, while the standard gradient is wrong
by 4.60e-08 — larger than the true gradient itself on the first entry.

### Grokking: `a + b mod 31` (paper Table 1)

1-layer, 4-head, width-64 ReLU Transformer, 10,000 full-batch AdamW
steps (lr 1e-3, betas 0.9/0.98, weight decay 1), 50% train split.
Test accuracy and mean train margin at the final step; spikes are
distinct slingshot events (loss > 10x running minimum after
memorisation); grokked means test accuracy first reached 0.99.

| loss | precision | eps | test acc mean | test acc min | final margin mean | equilibrium | spikes | grokked |
|---|---|---|---|---|---|---|---|---|
| fixed | fp32 | 1e-08 | 0.989 | 0.977 | 17.27 | 18.42 | 0 | 1/2 |
| fixed | bf16 | 1e-08 | 0.973 | 0.954 | 17.13 | 18.42 | 0 | 1/2 |
| fixed | fp32 | 1e-12 | 0.988 | 0.988 | 26.66 | 27.63 | 0 | 0/1 |
| standard | fp32 | 1e-08 | 0.906 | 0.852 | 16.28 | 18.42 | 12 | 0/2 |
| standard | bf16 | 1e-08 | 0.928 | 0.925 | 3.45 | 18.42 | 0 | 0/2 |
| standard | fp32 | 1e-12 | 0.929 | 0.929 | 15.56 | 27.63 | 7 | 0/1 |

Per-seed final test accuracy: fixed fp32 0.977 and 1.000 (the 1.000
seed grokked at step 9000); fixed bf16 0.954 and 0.992 (grok step
10000); standard fp32 0.960 and 0.852; standard bf16 0.925 and 0.931.
The fixed loss beats the standard loss on accuracy in every
precision/eps, reaches the margin equilibrium in both precisions, and
never spikes. The standard loss stalls at the cliff in fp32 (final
margins 16.1–16.4, just under the 16.6 cliff, with 12 spike events)
and collapses in bf16 (final margins 3.2–3.7 — below the 5.5 cliff,
it never even gets there). At eps 1e-12 the fixed margin climbs to
26.66, near the 27.63 equilibrium, while the standard margin is
still stuck at 15.56.

### Distillation: student starting at the teacher (paper Sec. 7)

Teacher: the fixed/fp32/seed-0 grokking run (test accuracy 0.977).
Student initialised at the teacher's exact weights, 2,000 full-batch
Adam steps (lr 2e-3, no weight decay) at temperature 4. The true
gradient is zero, so any KL growth or accuracy loss is rounding error
amplified by Adam.

| loss | precision | final KL | test acc | loss |
|---|---|---|---|---|
| standard | bf16 | 1.209e+02 | 0.854 | 6.130e+02 |
| standard | fp32 | 2.207e-05 | 0.977 | 2.618e-01 |
| fixed | bf16 | 0.000e+00 | 0.977 | 2.612e-01 |
| fixed | fp32 | 0.000e+00 | 0.977 | 2.618e-01 |

With the standard loss in bf16 the student drifts catastrophically
(KL 120.9, accuracy 0.977 -> 0.854); in fp32 it leaves a small KL
floor of 2.2e-05 with accuracy unchanged — the paper's "~1e-6 floor"
in the same ballpark. With the fixed loss the student does not move
at all (KL exactly 0) in either precision.

## Paper comparison

The paper's Table 1 (test accuracy on `a + b mod 31`, 3–5 seeds,
10,000 epochs): fp32 fixed 100.0 / 99.6 / 99.0 vs standard 94.3 /
96.4 / 96.5 at eps 1e-8 / 1e-12 / 1e-16; bf16 fixed 100.0 / 99.8 /
98.9 vs standard 93.3 / 94.9 / 94.2; final margins for the fixed loss
17.9 / 27.4 / 36.8, matching the equilibrium `log(1/eps)`. The
reproduction above uses the same setup with fewer seeds and eps
values (CPU budget — see `docs/RESEARCH.md`); the ordering and the
mechanism are what is being checked, and the magnitudes are reported
as measured, not matched to the paper.

## Limitations

- The Transformer here is a NumPy reimplementation, not the paper's
  JAX code: MLP width 256 and the `[a, b, =]` tokenisation are
  documented choices (the paper does not state them).
- Attention softmax inside the model stays in fp32; the fix applies
  to the *loss* softmax, as in the paper. Low-precision attention has
  its own failure mode (Qiu & Yao 2026) and is out of scope.
- bf16 arithmetic rounds the logits themselves, so bf16 gradients at
  large margins carry a few percent relative error (see above) — the
  fix removes the cliff, it does not repeal the dtype.
- The distillation teacher is the fixed/fp32 grokking run's model
  (test accuracy 0.977, not the paper's 100%), and the student is a
  1-layer Transformer rather than the paper's 2-layer MLP, because
  the student must share the teacher's architecture to start from its
  exact weights.

## Project layout

```
src/softmax_cliff/
  losses.py       standard/fixed log-softmax, softmax, hard + soft CE
  model.py        1-layer Transformer + AdamW (manual backprop)
  data.py         modular-arithmetic tasks and splits
  probes.py       exact cliff / gradient-sum / cancellation probes
  experiment.py   grokking + distillation training loops
scripts/
  demo.py             probes + short grokking run
  run_experiment.py   full grid -> results/experiment.json, measured.json
  crossfoot.py        README figures vs results
tests/                25 tests (paper examples, finite differences,
                      probes, determinism, no-branding scan)
docs/RESEARCH.md      trending check, paper notes, design decisions,
                      bugs found during the night
```

## Scale-up path

The loss functions are drop-in replacements: `hard_ce_fixed` /
`soft_ce_fixed` take logits and targets and return `(loss, grad)` in
the requested precision, so the same code ports to a JAX/PyTorch
training loop (the paper's own JAX snippets are the reference). For
real LLM training, apply the fixed loss at the lm-head in bf16 and
keep the rest of the pipeline unchanged; the paper reports no
meaningful slowdown. The grokking harness here is a CPU proxy — for
a real model, run the same comparison with TRL's GRPOTrainer or a
SageMaker training job at the target precision.

## References

- Barron, Gnanapragasam, Tschannen, "A Fix for Softmax and
  Cross-Entropy", arXiv:2610.31642, v3, 29 Oct 2026.
  https://arxiv.org/abs/2610.31642
- Prieto et al., "Grokking at the Edge of Numerical Stability",
  ICLR 2025 (Softmax Collapse; StableMax baseline).
- Liu et al., "Grokking or Glitching? How Low-Precision Drives
  Slingshot Loss Spikes", arXiv:2605.06152, 2026 (Zero-sum Proj.).
- Varma et al., "Explaining Grokking Through Circuit Efficiency",
  arXiv:2309.02390, 2023 (grokking setup).
- Thilak et al., "The Slingshot Mechanism", arXiv:2206.04817, 2022.
- Hinton et al., "Distilling the Knowledge in a Neural Network",
  arXiv:1503.02531, 2015.

## License

MIT — see `LICENSE`.
