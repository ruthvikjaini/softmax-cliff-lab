"""Run the full experiment grid and write results/*.json.

Grid (paper Sec. 6–7, Table 1, reduced to what a CPU can do in one
night — see docs/RESEARCH.md for the reduction rationale):

1. Grokking on ``a + b mod 31`` (paper Table 1): loss {standard, fixed}
   x precision {bf16, fp32} x eps {1e-8} x seeds {0, 1, 2}, plus
   eps {1e-12} x seeds {0} for both losses in fp32 (the margin
   equilibrium check, paper Eq. 11).
2. Distillation (paper Sec. 7): one teacher trained with the fixed
   loss to ~100% test accuracy; students distilled with the standard
   and fixed soft-target CE in bf16 and fp32 at temperature 4.
3. Exact probes (no training): cliff scan, gradient sums,
   cancellation scan, perturbed example.

Usage:
    python scripts/run_experiment.py [--quick] [--steps N] [--seeds 0 1]

``--quick`` runs a smoke grid (1 seed, 600 steps) for CI / sanity
checks and writes to results/experiment_quick.json instead of
overwriting the shipped results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from softmax_cliff.experiment import (  # noqa: E402
    train_distillation,
    train_grokking,
)
from softmax_cliff.probes import (  # noqa: E402
    cancellation_perturbed,
    cancellation_scan,
    cliff_scan,
    cliff_summary,
    equilibrium_margin,
    gradient_sum_scan,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _aggregate_grokking(rows):
    """Aggregate grokking runs into headline numbers per config."""
    groups = {}
    for row in rows:
        key = (row["loss"], row["precision"], row["eps"])
        groups.setdefault(key, []).append(row)
    out = []
    for (loss, precision, eps), group in sorted(groups.items()):
        accs = [g["final_test_acc"] for g in group]
        margins = [g["final_mean_margin"] for g in group]
        spikes = [g["n_spikes"] for g in group]
        groks = [g["grok_step"] for g in group if g["grok_step"] is not None]
        out.append({
            "loss": loss,
            "precision": precision,
            "eps": eps,
            "n_seeds": len(group),
            "final_test_acc_mean": float(np.mean(accs)),
            "final_test_acc_min": float(np.min(accs)),
            "final_test_acc_values": [round(a, 4) for a in accs],
            "final_mean_margin_mean": round(float(np.mean(margins)), 2),
            "final_mean_margin_values": [round(m, 2) for m in margins],
            "equilibrium_margin": round(equilibrium_margin(eps), 2),
            "n_spikes_total": int(sum(spikes)),
            "grok_step_median": int(np.median(groks)) if groks else None,
            "n_groked": len(groks),
        })
    return out


def _aggregate_distillation(rows):
    out = []
    for row in rows:
        out.append({
            "loss": row["loss"],
            "precision": row["precision"],
            "final_kl": row["final_kl"],
            "final_test_acc": row["final_test_acc"],
            "final_loss": row["final_loss"],
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.quick:
        steps = args.steps or 600
        seeds = args.seeds or [0]
        out_name = "experiment_quick.json"
    else:
        steps = args.steps or 10000
        seeds = args.seeds or [0, 1]
        out_name = "experiment.json"

    grokking_rows = []
    teacher_model = None
    print(f"=== grokking grid: {steps} steps, seeds {seeds} ===")
    for loss in ("standard", "fixed"):
        for precision in ("bf16", "fp32"):
            for seed in seeds:
                keep_model = (loss == "fixed" and precision == "fp32"
                              and seed == seeds[0])
                row = train_grokking(task="add", loss=loss,
                                     precision=precision, eps=1e-8,
                                     steps=steps, seed=seed, log_every=1000,
                                     return_model=keep_model)
                if keep_model:
                    teacher_model = row.pop("model")
                    teacher_result = {k: v for k, v in row.items()}
                grokking_rows.append(row)
                print(f"  {loss:>8} {precision:>4} eps=1e-8 seed={seed}: "
                      f"test_acc {row['final_test_acc']:.3f} "
                      f"margin {row['final_mean_margin']:.2f} "
                      f"grok_step {row['grok_step']} "
                      f"spikes {row['n_spikes']} "
                      f"({row['wall_seconds']}s)")
    # Epsilon sweep for the margin equilibrium (paper Eq. 11).
    for loss in ("standard", "fixed"):
        row = train_grokking(task="add", loss=loss, precision="fp32",
                             eps=1e-12, steps=steps, seed=seeds[0],
                             log_every=1000)
        grokking_rows.append(row)
        print(f"  {loss:>8} fp32 eps=1e-12 seed={seeds[0]}: "
              f"test_acc {row['final_test_acc']:.3f} "
              f"margin {row['final_mean_margin']:.2f} "
              f"(equilibrium {equilibrium_margin(1e-12):.1f})")

    print("=== distillation (paper Sec. 7) ===")
    # Reuse the fixed/fp32/seed-0 grokking run's model as the teacher:
    # it was trained with the fixed loss and reached ~100% test accuracy.
    print(f"  teacher: test_acc {teacher_result['final_test_acc']:.3f} "
          f"margin {teacher_result['final_mean_margin']:.2f}")
    distill_rows = []
    for loss in ("standard", "fixed"):
        for precision in ("bf16", "fp32"):
            row = train_distillation(teacher_model, task="add", loss=loss,
                                     precision=precision, steps=2000,
                                     seed=0, log_every=500)
            distill_rows.append(row)
            print(f"  {loss:>8} {precision:>4}: final KL "
                  f"{row['final_kl']:.3e} test_acc "
                  f"{row['final_test_acc']:.3f}")

    print("=== exact probes ===")
    cliff_rows = cliff_scan()
    probe_results = {
        "cliff_summary": cliff_summary(cliff_rows),
        "gradient_sum": gradient_sum_scan(),
        "cancellation": cancellation_scan(),
        "perturbed_example": cancellation_perturbed(
            [2.0, 1.0, 0.0], [-1e-7, 1e-7, 0.0], precision="fp32"),
    }
    for row in probe_results["cliff_summary"]:
        print(f"  cliff {row['precision']:>5} {row['loss']:<8}: "
              f"{row['cliff']} (paper {row['paper_cliff']})")

    experiment = {
        "grokking": grokking_rows,
        "teacher": teacher_result,
        "distillation": distill_rows,
        "probes": probe_results,
    }
    measured = {
        "grokking": _aggregate_grokking(grokking_rows),
        "distillation": _aggregate_distillation(distill_rows),
        "probes": probe_results,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / out_name).write_text(json.dumps(experiment, indent=2))
    if not args.quick:
        (RESULTS / "measured.json").write_text(json.dumps(measured, indent=2))
    print(f"wrote {RESULTS / out_name}"
          + ("" if args.quick else " and results/measured.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
