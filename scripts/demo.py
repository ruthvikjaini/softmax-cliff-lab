"""Quick demo: the two bugs, measured in seconds.

Runs the exact probes (precision cliff, gradient sum, cancellation
residual) and a short grokking run, then prints the headline numbers.
No training grid, no file writes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from softmax_cliff.experiment import train_grokking  # noqa: E402
from softmax_cliff.probes import (  # noqa: E402
    cancellation_perturbed,
    cancellation_scan,
    cliff_scan,
    cliff_summary,
    gradient_sum_scan,
)


def main():
    print("=== Softmax Collapse: the precision cliff ===")
    rows = cliff_scan()
    for row in cliff_summary(rows):
        print(f"  {row['precision']:>5} {row['loss']:<8}: first zero target "
              f"gradient at margin {row['cliff']} (paper: "
              f"{row['paper_cliff']})")
    print("  (margins in the probe are exact; the paper's numbers come from "
          "its own logit construction)")

    print("\n=== Gradient sums at margin 20 (exact value: 0) ===")
    for row in gradient_sum_scan(margins=(20.0,)):
        print(f"  {row['precision']:>5} {row['loss']:<8} grad sum "
              f"{row['grad_sum']:.3e}  (target grad {row['grad_t']:.3e}, "
              f"exact {row['exact_abs_grad_t']:.3e})")

    print("\n=== Softmax Cancellation: residual gradient at z_S == z_T ===")
    for row in cancellation_scan(n_teachers=200):
        print(f"  {row['precision']:>5} standard median |g| "
              f"{row['standard_median']:.2e} (max {row['standard_max']:.2e}); "
              f"fixed max |g| {row['fixed_max']:.1e}")

    print("\n=== Paper's perturbed example (fp32) ===")
    out = cancellation_perturbed([2.0, 1.0, 0.0], [-1e-7, 1e-7, 0.0])
    print(f"  exact    {['%.2e' % v for v in out['exact']]}")
    print(f"  standard {['%.2e' % v for v in out['standard']]} "
          f"(max err {out['standard_max_err']:.2e})")
    print(f"  fixed    {['%.2e' % v for v in out['fixed']]} "
          f"(max err {out['fixed_max_err']:.2e})")

    print("\n=== Short grokking run: a + b mod 31, fixed CE, fp32 ===")
    result = train_grokking(task="add", loss="fixed", precision="fp32",
                            steps=600, seed=0, log_every=200)
    for row in result["trajectory"]:
        print(f"  step {row['step']:>5}: loss {row['train_loss']:.4f}  "
              f"train acc {row['train_acc']:.3f}  "
              f"test acc {row['test_acc']:.3f}  "
              f"mean margin {row['mean_margin']:.2f}")
    print("  (full grid: python scripts/run_experiment.py)")


if __name__ == "__main__":
    main()
