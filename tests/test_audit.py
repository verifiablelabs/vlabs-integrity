"""Gameability audit — a weak (single-test) reward is certified gameable; a strong one is not."""
from __future__ import annotations

from vlabs_integrity import audit_reward
from vlabs_integrity.verifiers_adapter import build_integrity_rubric, integrity_reward_fn

REF = (
    "def solve(a):\n"
    "    n = 0\n"
    "    for x in a:\n"
    "        if x > 0:\n"
    "            n = n + 1\n"
    "    return n\n"
)


def _gen(i):
    return ([i % 5, (i * 3) % 7, (i * 2) % 4, i % 6],)


def test_single_provided_test_leaves_exploitable_gap():
    # one all-positive provided test misses the >/>= boundary mutant that the isomorphic
    # re-grade (fresh inputs incl. zeros) catches -> a real exploitable gap.
    r = audit_reward(REF, "solve", [([1, 2, 3],)], _gen)
    assert r.n_mutants > 0
    assert 0.0 <= r.deployed_kill_rate <= 1.0
    assert r.trusted_kill_rate >= r.deployed_kill_rate  # isomorphic bank catches at least as many
    assert r.exploitable_gap > 0.0  # the provided reward misses part of the surface
    assert r.isomorphic_shortcut is True


def test_rich_isomorphic_bank_closes_the_gap():
    r_weak = audit_reward(REF, "solve", [([1, 2, 3],)], _gen, n_isomorphic=0)
    r_strong = audit_reward(REF, "solve", [([1, 2, 3],)], _gen, n_isomorphic=24)
    # more isomorphic inputs cannot lower the trusted kill rate
    assert r_strong.trusted_kill_rate >= r_weak.trusted_kill_rate


def test_report_serialises_and_is_honest():
    r = audit_reward(REF, "solve", [([1, 2, 3],), ([4, 5],)], _gen)
    d = r.to_dict()
    import json

    json.loads(json.dumps(d))
    assert any("public method" in h for h in d["honesty"])
    assert any("does not prove the verifier unhackable" in h for h in d["honesty"])


def test_verifiers_reward_fn_is_deterministic_and_bounded():
    fn = integrity_reward_fn(REF, "solve", [([1, 2, 3],)], _gen)
    v1 = fn("prompt", "completion", "answer")
    v2 = fn("other", "rollout", None)
    assert v1 == v2  # deterministic, ignores the rollout
    assert 0.0 <= v1 <= 1.0
    assert fn.__name__ == "reward_integrity"


def test_build_rubric_requires_verifiers_extra():
    # the verifiers package is an optional extra; without it, a helpful RuntimeError
    import pytest

    try:
        import verifiers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="optional 'verifiers'"):
            build_integrity_rubric(REF, "solve", [([1, 2, 3],)], _gen)
