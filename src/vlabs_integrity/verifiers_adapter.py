"""Adapter for the `verifiers` framework (Prime Intellect Environments Hub).

`verifiers` scores rollouts with a `Rubric` of reward functions. Add this as an extra reward
function to surface, alongside your task reward, WHETHER the task's reward is gameable — so a
gap between "passes the provided tests" and "passes the isomorphic re-grade" is visible in the
env's own evaluation.

The `verifiers` import is LAZY and OPTIONAL: this module works standalone, and only touches
`verifiers` when you actually build a rubric with it. Adapt `extract_*` to your env's task shape.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from . import audit_reward


def integrity_reward_fn(
    reference_source: str,
    entry_point: str,
    provided_inputs: Sequence[tuple],
    input_generator: Callable[[int], tuple],
    *,
    penalize_gap: float = 1.0,
) -> Callable[..., float]:
    """Return a `verifiers`-style reward function that scores REWARD INTEGRITY for a fixed task.

    The returned callable ignores the rollout and returns ``1 - penalize_gap * exploitable_gap``
    for the task's reward: 1.0 when the deployed reward matches the trusted isomorphic re-grade,
    lower when it leaves an exploitable surface. Use it as an auxiliary rubric signal / gate, not
    as the task reward itself.
    """
    report = audit_reward(reference_source, entry_point, provided_inputs, input_generator)
    score = max(0.0, 1.0 - penalize_gap * report.exploitable_gap)

    def _reward(*_args, **_kwargs) -> float:  # verifiers passes (prompt, completion, answer, ...)
        return score

    _reward.__name__ = "reward_integrity"
    return _reward


def build_integrity_rubric(*args, **kwargs):  # pragma: no cover - requires the verifiers extra
    """Wrap :func:`integrity_reward_fn` in a `verifiers.Rubric` (requires `pip install verifiers`)."""
    try:
        import verifiers as vf  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "build_integrity_rubric needs the optional 'verifiers' package: pip install verifiers"
        ) from e
    return vf.Rubric(funcs=[integrity_reward_fn(*args, **kwargs)], weights=[1.0])


__all__ = ["build_integrity_rubric", "integrity_reward_fn"]
