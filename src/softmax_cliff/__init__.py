"""Reproduction and extension of Barron et al., "A Fix for Softmax and
Cross-Entropy" (arXiv:2610.31642, 2026): the two numerical bugs in
standard softmax / cross-entropy, their fixes, and what they do to
grokking and distillation.
"""

from . import data, experiment, losses, model, probes

__all__ = ["data", "experiment", "losses", "model", "probes"]
__version__ = "0.1.0"
