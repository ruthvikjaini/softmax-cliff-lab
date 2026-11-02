# Research notes — 2026-11-02 (Day 9)

## Trending-first check (before any paper research)

GitHub trending, daily and weekly, checked 2026-11-02 ~00:00 ET:

- Daily top: thedotmack/claude-mem (persistent agent context),
  lightpanda-io/browser (headless browser for AI/automation),
  shiyu-coder/Kronos (foundation model for financial markets),
  pingdotgg/t3code, milind-soni/OpenMausBot, public-apis/public-apis,
  cjpais/Handy (offline speech-to-text), Donchitos/Claude-Code-Game-Studios,
  kornia/kornia (geometric computer vision), beenuar/AiSOC (AI security
  operations centre), CopilotKit/OpenBot, FlashML-org/FreeToken
  (datacenter-scale model serving on desktop), hyprwm/Hyprland.
- Weekly top: Nutlope/hallmark, pbakaus/impeccable, AgriciDaniel/claude-seo,
  thedotmack/claude-mem, cloudflare/security-audit-skill (56,550 stars,
  3,686 forks, 10,069 stars this week), obra/superpowers, JetBrains/thinkrail,
  pingdotgg/t3code, multica-ai/andrej-karpathy-skills, jakubkrehel/skills
  (10,142 stars, 394 forks), TauricResearch/TradingAgents, anthropics/skills,
  affaan-m/ECC, ghostty-org/ghostty.

Verdict: same as Days 5–8. The trending list is again dominated by
prompt/skill packs (claude-mem, hallmark, impeccable, superpowers, ECC,
security-audit-skill, claude-seo, karpathy-skills), agent harnesses and
full platforms (OpenBot, TradingAgents, DeepTutor from prior nights).
None has a benchmarkable code weakness worth optimizing: skill packs are
prose, not code; the platforms are whole products, not a measurable
component. The code repos that do appear (Kronos, FreeToken, kornia,
lightpanda browser) are either foundation models needing GPU training,
serving systems needing GPU hardware, or mature libraries with no
measurable weakness a one-night CPU build could honestly improve.
No trending repo was picked. Fell back to the paper-based original
project, per the standing rule.

## Primary source (verified)

Barron, Gnanapragasam, Tschannen (Google DeepMind), "A Fix for Softmax
and Cross-Entropy", arXiv:2610.31642, submitted 22 Oct 2026, v3 revised
29 Oct 2026. Verified on the arXiv abstract page and the full HTML on
2026-11-02. This paper was explicitly held for a later night in the
Day 6 log ("needs bf16 training harness") — this is that night.

Key facts used (all from the paper, not secondary summaries):

- Two bugs in the standard max-shift implementations used by every
  framework (JAX, PyTorch, TensorFlow). Bug 1, Softmax Collapse
  (Sec. 2): once the logit margin `m = z_t - logsumexp_{j != t} z_j`
  exceeds the precision cliff `c_d` (~5.5 bf16, ~16.6 fp32, ~36.7 fp64),
  the denominator `1 + exp(-m)` rounds to 1, the target gradient
  `q_t - 1` rounds to zero while the non-target gradients stay positive,
  so the gradients no longer sum to zero. Bug 2, Softmax Cancellation
  (Sec. 3): in soft-target distillation, `q - p` is computed by
  subtracting two nearly-equal probabilities evaluated through different
  op orders, leaving a non-zero residual gradient even when
  `z_S == z_T` (~1e-3 bf16, ~1e-8 fp32, ~1e-17 fp64, Fig. 1(b)).
- Fix (Eq. 5–7, 9): never add 1 to `exp(-m)` — set the argmax/target
  slot to exactly 0, exclude it from the sum of exponentials, use
  `softplus(-m)` / `log1p` for the loss, and take the target gradient as
  the negated sum of the non-target gradients; for soft targets compute
  `q - p` in log space from `delta = z_S - z_T` via `expm1`/`log1p`,
  centred under `p`. Gradients stay accurate down to exponent
  underflow and sum to zero by construction.
- Worked examples used as test fixtures: `z = [9, -9, -9]`, `t = 0`
  (standard: loss 0, grad `[0, 1.523e-8, 1.523e-8]`; fixed: loss
  3.046e-8, grad `[-3.046e-8, 1.523e-8, 1.523e-8]`); perturbed
  distillation `z_T = [2, 1, 0]`, `z_S = z_T + [-1e-7, 1e-7, 0]`
  (exact grad `[-4.596e-8, 4.144e-8, 4.513e-9]`).
- Mechanism (Sec. 4): after memorisation, Adam's second moment decays
  while examples sit above the cliff; when weight decay pulls examples
  back below the cliff their target gradients switch back on and Adam
  divides by a decayed denominator — the "slingshot" loss spikes.
- Margin equilibrium (Sec. 5, Eq. 11): with weight decay and AdamW,
  margins saturate at `m*(eps) = log(1/eps)` — 18.4 for eps=1e-8,
  27.6 for 1e-12, 36.8 for 1e-16. The default eps=1e-8 puts the
  equilibrium just above the fp32 cliff (16.6), which is why standard
  CE spikes in fp32.
- Grokking experiments (Sec. 6, Table 1): 1-layer, 4-head, width-64
  ReLU Transformer without LayerNorm on `a + b mod 31` (50% train
  split), 10,000 epochs full-batch AdamW (lr 1e-3, betas 0.9/0.98,
  weight decay 1), eps in {1e-8, 1e-12, 1e-16}, logit precision at the
  loss in {bf16, fp32, fp64}. Table 1 (test accuracy, 3–5 seeds):
  fp32 fixed 100.0 / 99.6 / 99.0 vs standard 94.3 / 96.4 / 96.5;
  bf16 fixed 100.0 / 99.8 / 98.9 vs standard 93.3 / 94.9 / 94.2;
  final margins fixed 17.9 / 27.4 / 36.8 vs standard 13.0 / 15.1 / 12.7.
  Baselines: Zero-sum Proj. (Liu et al. 2026) and StableMax (Prieto et
  al. 2025) both lag fixed CE. Fig. 6 extends to `a + b mod 113` and
  2-layer MLPs on `a^2 + b^2 mod 97` and `a^3 + b mod 97`.
- Distillation (Sec. 7): 2-layer ReLU MLP student initialised with the
  teacher's weights (100% on `a + b mod 53`), 2,000 full-batch Adam
  steps (lr 2e-3, betas 0.9/0.98, no weight decay, eps 1e-8) at
  temperature 4. Standard soft-target CE drifts the student away in
  bf16 and leaves a ~1e-6 KL floor in fp32; fixed stays at KL = 0 and
  100% accuracy in all precisions.
- Errata (v3): an earlier version's code produced NaN gradients when
  all but one logit was effectively -inf (e.g. attention masking); the
  v3 code uses `log1p(s)` rather than `softplus(log s)`. This
  implementation uses `log1p` throughout and handles the degenerate
  case (see `losses.py`).

## Other papers considered and rejected

- Liu, "There Is No Free Lunch in RLVR" (arXiv:2610.39295, 29 Oct
  2026): a theorem that every pass@1 gain has a pass@k price, with a
  verification package. Strong and very current, but it is a theory
  paper in the same RLVR cluster as Day 1 (GRPO) and Day 7 (score
  centering); the Barron paper opens a different cluster (training
  numerics) and was already queued from Day 6. Held as a candidate for
  a later night.
- Iyengar, "Sample, Mix, or Verify?" (arXiv:2610.23255, 16 Oct 2026)
  and Bilal et al., "Adaptive Replay Refinement" (arXiv:2610.28294,
  20 Oct 2026): test-time compute allocation under imperfect verifiers
  / revision-vs-resampling. Both overlap Day 6 (PORTAL test-time
  intervention) too closely.
- Illouz, "Resample the Answer" (arXiv:2610.16427, v2 26 Oct 2026):
  Gated Answer Resampling for the Tiny Recursive Model (85.9% ->
  99.4% on Sudoku-Extreme). Interesting, but needs a recursive-model
  implementation and puzzle benchmarks; heavier than one CPU night
  allows honestly.
- CMU/Hopkins unified training (29 Oct 2026, noted in AI Weekly):
  folds pretrain/SFT/RL into one loss; needs real small-model runs
  (124M params), too large for one honest CPU night (same verdict as
  Day 7).

## Implementation choices and simulation assumptions

- NumPy only (+ `ml_dtypes` for real bfloat16 arithmetic). No JAX /
  PyTorch: the loss functions must be evaluated in genuine bf16/fp32/
  fp64 arithmetic, and `ml_dtypes.bfloat16` gives that inside NumPy
  ufuncs. No GPU, no LLM calls, no paid APIs.
- The paper's JAX code is used only as the formula reference for
  Eq. 5–7 and 9; all code here is an independent NumPy implementation
  with hand-written gradients, verified against finite differences
  and fp64 references in `tests/`.
- Transformer: 1 layer, 4 heads, width 64, ReLU MLP (hidden 256), no
  LayerNorm, as in the paper. The paper does not state the MLP width
  or the tokenisation, so those are documented choices: input
  `[a, b, =]` (3 tokens, separator id P), readout from the final
  position, learned token + positional embeddings, Xavier-uniform
  linears. Initialisation: embeddings N(0, 1/sqrt(d)).
- Logit precision at the loss: weights stay fp32; logits are cast to
  the target dtype for the loss, and the loss gradient is cast back
  to fp32 for the backward pass — the paper's setup.
- Attention softmax inside the model stays in fp32: the paper's fix
  concerns the *loss* softmax, not attention. (Attention softmax has
  its own low-precision failure mode — Qiu & Yao 2026, cited in the
  paper — but that is out of scope here.)
- Standard soft-target CE reproduces the paper's forward/backward
  op-order mismatch deliberately: `p` from the forward-pass softmax,
  `q` from `exp` of the backward-path log-softmax.
- Grid reduction vs the paper: the paper trains 10,000 epochs x
  3–5 seeds x 3 eps x 3 precisions x 2 losses. On CPU NumPy one
  10,000-step run takes ~12 minutes, so the grid here is loss
  {standard, fixed} x precision {bf16, fp32} x eps {1e-8} x seeds
  {0, 1} at 10,000 steps, plus eps {1e-12} x 1 seed in fp32 for the
  equilibrium check. fp64 grokking is omitted (its cliff, 36.7, is
  above the eps=1e-8 equilibrium 18.4, so the paper's own Table 1
  shows standard fp64 already at 100% there). The probes (cliff,
  cancellation) cover all three precisions exactly.
- Distillation reuses the fixed/fp32/seed-0 grokking run's model as
  the teacher (it reached ~100% test accuracy), instead of training a
  separate teacher as the paper does; the student is a 1-layer
  Transformer here rather than the paper's 2-layer MLP, because the
  student must share the teacher's architecture to start from its
  exact weights. Temperature 4, 2,000 steps, lr 2e-3, as in the paper.
- Spike counting: a slingshot *event* is a transition into loss >
  10x the running minimum after memorisation (loss < 0.01), with
  hysteresis (the episode ends only when the loss falls back below
  2x the running minimum), so a rapidly oscillating loss counts as
  one sustained episode, not hundreds of events.

## Bugs found and fixed during the night

1. `model.backward`: `W_2` gradient used `h.T @ d_mlp_out` with
   mismatched shapes — `h` is (N, T, d_mlp) and must be flattened
   over batch x time first. Caught by the gradient check (shape
   error), fixed before any training ran.
2. `model.backward`: the embedding gradient used `np.add.at`, which
   profiled at 21 ms/step — 23% of the step time. Replaced with a
   one-hot matmul (same result, verified by the finite-difference
   tests). Step time 0.093 s -> ~0.072 s.
3. `experiment._spike_count` first counted every spiked *step*
   (1,806 in the standard pilot run); redefined to count distinct
   spike events with hysteresis.
4. Test expectations, not code: the paper's "exact" soft-target
   gradient values are exact w.r.t. the fp32-representable logits
   (the 1e-7 perturbation itself rounds in fp32), and the fixed
   bf16 gradient at large margins carries ~5–6% relative error
   because subtracting two bf16 logits of magnitude ~20 loses ~0.06
   absolute precision — a property of the dtype, not of the fix.
   Both tests now compare against the exact gradient of the
   representable logits, with a 10% bf16 tolerance documented in
   the test.
5. Pilot finding (kept, not a bug): a 2-layer MLP on `a + b mod 31`
   memorises (train loss -> 0) but does not grok within 3,000 steps
   (test accuracy ~0.02); the Transformer is required, as in the
   paper. The MLP pilot was discarded, not shipped.
6. Pilot finding: at 4,000 steps neither loss has grokked yet
   (test accuracy ~0.40 both); the accuracy separation in the
   paper's Table 1 needs the full 10,000 epochs. The margin and
   spike separation is already visible at 4,000 steps (fixed margin
   climbs smoothly to 16.6; standard margin oscillates 5.7–15.9
   with loss spikes to 4e-3).

## Verification performed

- Paper worked examples reproduced exactly (losses and gradients,
  Sec. 2–3) in `tests/test_losses.py`.
- Transformer backprop verified against central finite differences
  (max error 1.5e-8, the fp32 cast floor) in `tests/test_model.py`.
- Cliff scan, gradient-sum scan, cancellation scan reproduce the
  paper's Figure 1 magnitudes; see `results/measured.json`.
- Full grid rerun determinism is asserted for a short run in
  `tests/test_experiment.py` (two identical 300-step runs).
- The full grid ran 2026-11-02 (~2h20m wall, single process, CPU
  contention from other tenants slowed it). The fixed/fp32/seed-0
  grokking run reproduced the earlier 10,000-step pilot exactly
  (test accuracy 0.9771309771309772, margin 17.436221970620327),
  confirming determinism across processes. The first grid run
  (standard/bf16/seed 0) took 1,334 s vs ~700 s for the others —
  wall-clock contention while other jobs overlapped, not a code
  difference; its results are unaffected (deterministic given the
  seed).
- Post-run fix: the diagnostic field `max_loss_after_memorisation`
  was actually the max train loss over all steps, so it was renamed
  to `max_train_loss` in `experiment.py` and in the shipped
  `results/experiment.json` (11 rows, values unchanged). It is not
  used in any aggregate or README figure.

## Measured results (full grid, 2026-11-02)

Grokking, `a + b mod 31`, 10,000 steps, 2 seeds unless noted
(test accuracy / final mean margin, equilibrium `log(1/eps)`):

- fixed fp32 eps 1e-8: 0.989 mean (0.977, 1.000), margin 17.27
  (eq 18.42), 0 spikes, 1/2 grokked (seed 1 at step 9000).
- fixed bf16 eps 1e-8: 0.973 mean (0.954, 0.992), margin 17.13,
  0 spikes, 1/2 grokked (step 10000).
- fixed fp32 eps 1e-12 (1 seed): 0.988, margin 26.66 (eq 27.63),
  0 spikes.
- standard fp32 eps 1e-8: 0.906 mean (0.960, 0.852), margin 16.28,
  12 spikes, 0/2 grokked.
- standard bf16 eps 1e-8: 0.928 mean (0.925, 0.931), margin 3.45
  (below the 5.5 bf16 cliff — it never gets there), 0 spikes.
- standard fp32 eps 1e-12 (1 seed): 0.929, margin 15.56 (eq 27.63),
  7 spikes.

Ordering matches the paper's Table 1 on every comparison (fixed
beats standard at every precision/eps; margins reach the equilibrium
only for the fixed loss), with magnitudes close to the paper's
(fp32 fixed 100.0 vs standard 94.3; here 98.9 vs 90.6 — the gap is
wider here because one standard seed was badly damaged by slingshots,
0.852). The fixed loss also halves the run-to-run spread.

Distillation, teacher = fixed/fp32/seed-0 grokking run (test acc
0.977), 2,000 steps at T=4:

- standard bf16: final KL 120.9, test acc 0.977 -> 0.854 (the student
  is driven away from the teacher, as in the paper).
- standard fp32: final KL 2.2e-05, test acc unchanged at 0.977
  (the paper's "~1e-6 floor" in the same ballpark).
- fixed bf16 / fp32: final KL exactly 0.0, test acc unchanged.

Exact probes: cliffs measured at 5.0 / 16.0 / 36.25 (paper 5.5 /
16.6 / 36.7, from a different logit construction); cancellation
residual medians 9.77e-04 / 2.98e-08 / 2.78e-17 (paper ~1e-3 /
~1e-8 / ~1e-17); fixed gradient matches the fp64 exact gradient to
1.03e-15 on the paper's perturbed example.
