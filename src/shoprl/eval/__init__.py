"""Rollout evaluation and the official ShopSimulator metrics."""

from shoprl.eval.metrics import outcome, summarise
from shoprl.eval.run import EvalConfig, evaluate

__all__ = ["EvalConfig", "evaluate", "outcome", "summarise"]

