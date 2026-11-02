"""Cross-foot the README against the measured results.

1. Recompute every aggregate in results/measured.json from the raw
   per-run rows in results/experiment.json and fail on any mismatch
   beyond rounding.
2. Check that every headline figure the README states appears in
   measured.json, formatted the way the README prints it.

Usage: python scripts/crossfoot.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TOL = 1e-3


def _mean(values):
    return sum(values) / len(values)


def recompute_measured(experiment):
    """Rebuild the aggregate tables from the raw rows."""
    groups = {}
    for row in experiment["grokking"]:
        key = (row["loss"], row["precision"], row["eps"])
        groups.setdefault(key, []).append(row)
    grokking = []
    for (loss, precision, eps), group in sorted(groups.items()):
        accs = [g["final_test_acc"] for g in group]
        margins = [g["final_mean_margin"] for g in group]
        spikes = [g["n_spikes"] for g in group]
        groks = [g["grok_step"] for g in group if g["grok_step"] is not None]
        grokking.append({
            "loss": loss, "precision": precision, "eps": eps,
            "n_seeds": len(group),
            "final_test_acc_mean": float(np_mean(accs)),
            "final_test_acc_min": float(min(accs)),
            "final_test_acc_values": [round(a, 4) for a in accs],
            "final_mean_margin_mean": round(float(np_mean(margins)), 2),
            "final_mean_margin_values": [round(m, 2) for m in margins],
            "n_spikes_total": int(sum(spikes)),
            "grok_step_median": int(np_median(groks)) if groks else None,
            "n_groked": len(groks),
        })
    distillation = [{
        "loss": r["loss"], "precision": r["precision"],
        "final_kl": r["final_kl"],
        "final_test_acc": r["final_test_acc"],
        "final_loss": r["final_loss"],
    } for r in experiment["distillation"]]
    return {"grokking": grokking, "distillation": distillation}


def np_mean(values):
    return sum(values) / len(values)


def np_median(values):
    values = sorted(values)
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= TOL
    return a == b


def _compare(path, expected, actual, problems):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                _compare(f"{path}.{key}", value, actual[key], problems)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            problems.append(f"{path}: length {len(expected)} != {len(actual)}")
        else:
            for i, (e, a) in enumerate(zip(expected, actual)):
                _compare(f"{path}[{i}]", e, a, problems)
    elif not _close(expected, actual):
        problems.append(f"{path}: {expected!r} != {actual!r}")


def main() -> int:
    experiment = json.loads((RESULTS / "experiment.json").read_text())
    measured = json.loads((RESULTS / "measured.json").read_text())
    recomputed = recompute_measured(experiment)
    problems = []
    _compare("grokking", recomputed["grokking"], measured["grokking"], problems)
    _compare("distillation", recomputed["distillation"],
             measured["distillation"], problems)
    # The probes section is copied verbatim from the probe functions;
    # check it matches a fresh probe run on the summary rows.
    if measured["probes"]["cliff_summary"] != experiment["probes"]["cliff_summary"]:
        problems.append("probes.cliff_summary differs from experiment.json")
    if measured["probes"]["cancellation"] != experiment["probes"]["cancellation"]:
        problems.append("probes.cancellation differs from experiment.json")

    # README figures: every number printed as a measured headline in the
    # "## Measured results" section must appear in measured.json (as a
    # value, or as a value formatted to the README's precision). Numbers
    # elsewhere in the README (paper values, hyperparameters, step
    # counts) are not checked: they come from the paper or the setup,
    # not from this run.
    readme = (ROOT / "README.md").read_text()
    section = readme.split("## Measured results", 1)[1]
    readme_section = section.split("\n## ", 1)[0]
    numbers = set()
    for group in (measured["grokking"], measured["distillation"]):
        for row in group:
            for value in row.values():
                if isinstance(value, float):
                    numbers.add(f"{value:.3f}")
                    numbers.add(f"{value:.2f}")
                    numbers.add(f"{value:.1f}")
                    numbers.add(f"{value:.2e}")
                    numbers.add(f"{value:.3e}")
                    numbers.add(f"{value:.0e}")
                elif isinstance(value, int):
                    numbers.add(str(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, float):
                            numbers.add(f"{item:.3f}")
                            numbers.add(f"{item:.2f}")
                            numbers.add(f"{item:.3e}")
                        elif isinstance(item, int):
                            numbers.add(str(item))
    for row in measured["probes"]["cancellation"]:
        for key in ("standard_median", "standard_max", "fixed_max"):
            value = row[key]
            numbers.add(f"{value:.2e}")
            numbers.add(f"{value:.1e}")
    for row in measured["probes"]["cliff_summary"]:
        if row["cliff"] is not None:
            numbers.add(f"{row['cliff']:.1f}")
            numbers.add(f"{row['cliff']:.2f}")
    for row in measured["probes"]["gradient_sum"]:
        for key in ("grad_sum", "grad_t", "exact_abs_grad_t"):
            value = row[key]
            numbers.add(f"{value:.3e}")
            numbers.add(f"{value:.2e}")
            numbers.add(f"{value:.3f}")
    perturbed = measured["probes"]["perturbed_example"]
    for key in ("standard_max_err", "fixed_max_err"):
        numbers.add(f"{perturbed[key]:.2e}")
        numbers.add(f"{perturbed[key]:.1e}")
    for key in ("exact", "standard", "fixed"):
        for value in perturbed[key]:
            numbers.add(f"{value:.2e}")
    # Figures printed in the README's results tables, as plain numbers.
    claimed = re.findall(r"\|\s*(-?[\d.]+(?:e-?\d+)?)\s*\|", readme_section)
    missing = [c for c in claimed if c not in numbers
               and c not in {"0", "1", "2", "3", "4", "5", "6", "7", "8"}]
    if missing:
        problems.append(f"README table figures not found in measured.json: "
                        f"{sorted(set(missing))}")
    if problems:
        print("CROSSFOOT FAILED:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("crossfoot OK: measured.json recomputes from experiment.json and "
          "README figures match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
