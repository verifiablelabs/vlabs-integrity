"""Gameability audit — a weak (single-test) reward is certified gameable; a strong one is not."""
from __future__ import annotations

import math

import pytest

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
    r_weak = audit_reward(REF, "solve", [([1, 2, 3],)], _gen, n_isomorphic=1)
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


def test_zero_mutants_fail_closed_and_score_zero():
    reference = "def solve(value):\n    return value\n"
    report = audit_reward(reference, "solve", [(1,)], lambda i: (i + 1,))

    assert report.n_mutants == 0
    assert report.auditable is False
    assert report.gameable is True
    assert report.deployed_kill_rate == 0.0
    assert report.trusted_kill_rate == 0.0
    assert report.failure_reason == "no_semantics_altering_mutants"

    reward = integrity_reward_fn(reference, "solve", [(1,)], lambda i: (i + 1,))
    assert reward("prompt", "completion", None) == 0.0


def test_reference_and_mutants_never_execute_in_parent(monkeypatch):
    import os

    marker = "VLABS_INTEGRITY_PARENT_EXEC_TEST"
    monkeypatch.delenv(marker, raising=False)
    reference = (
        "import os\n"
        f"os.environ[{marker!r}] = 'child-only'\n"
        "def solve(value):\n"
        "    return value + 1\n"
    )

    report = audit_reward(reference, "solve", [(1,)], lambda i: (i + 2,), n_isomorphic=2)

    assert report.n_mutants > 0
    assert os.environ.get(marker) is None


def test_isolated_runner_reaps_owned_process_group(monkeypatch):
    import vlabs_integrity

    calls = []
    monkeypatch.setattr(vlabs_integrity, "_kill_group", lambda proc: calls.append(proc.pid))

    outputs = vlabs_integrity._run_source("def solve():\n    return 1\n", "solve", [()])

    assert outputs == [1]
    assert calls


def test_nondeterministic_reference_fails_closed() -> None:
    reference = (
        "import os\n"
        "def solve(_value):\n"
        "    return os.urandom(16).hex()\n"
    )

    report = audit_reward(reference, "solve", [(1,)], lambda i: (i + 1,), n_isomorphic=2)

    assert report.auditable is False
    assert report.failure_reason == "nondeterministic_reference"
    assert report.n_mutants == 0
    assert report.deployed_kill_rate == 0.0
    assert report.trusted_kill_rate == 0.0
    assert report.gameable is True


def test_duplicate_only_generator_is_inconclusive_and_reward_zero() -> None:
    reference = "def solve(value):\n    return 1 if value > 0 else 0\n"
    report = audit_reward(
        reference,
        "solve",
        [(1,)],
        lambda _i: (1,),
        n_isomorphic=4,
    )
    assert report.auditable is False
    assert report.failure_reason == "insufficient_fresh_isomorphic_inputs"
    assert report.requested_isomorphic == 4
    assert report.achieved_isomorphic == 0
    assert report.gameable is True
    reward = integrity_reward_fn(reference, "solve", [(1,)], lambda _i: (1,))
    assert reward("prompt", "completion", None) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_isomorphic": 0}, "n_isomorphic"),
        ({"max_mutants": 0}, "max_mutants"),
        ({"weak_threshold": -0.1}, "weak_threshold"),
        ({"weak_threshold": float("nan")}, "weak_threshold"),
    ],
)
def test_audit_validates_coverage_and_threshold_parameters(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        audit_reward(REF, "solve", [([1, 2, 3],)], _gen, **kwargs)


def test_adapter_zeroes_gameable_even_when_exploitable_gap_is_zero() -> None:
    reference = "def solve(value):\n    return 1 if value > 0 else 0\n"

    def generator(i):
        return (i + 1,)

    report = audit_reward(reference, "solve", [(1,)], generator, n_isomorphic=4)
    assert report.auditable is True
    assert report.gameable is True
    assert report.exploitable_gap == 0.0
    reward = integrity_reward_fn(reference, "solve", [(1,)], generator)
    assert reward("prompt", "completion", None) == 0.0


@pytest.mark.parametrize("penalty", [-1.0, float("nan"), float("inf")])
def test_adapter_rejects_invalid_gap_penalty(penalty) -> None:
    assert not (math.isfinite(penalty) and penalty >= 0.0)
    with pytest.raises(ValueError, match="penalize_gap"):
        integrity_reward_fn(
            REF,
            "solve",
            [([1, 2, 3],)],
            _gen,
            penalize_gap=penalty,
        )


def test_runner_preserves_nested_tuple_and_list_types() -> None:
    import vlabs_integrity

    source = (
        "def solve(value):\n"
        "    return (isinstance(value, list), "
        "isinstance(value[0], tuple), "
        "isinstance(value[1]['nested'], tuple))\n"
    )
    outputs = vlabs_integrity._run_source(
        source,
        "solve",
        [([(1, 2), {"nested": (3, [4])}],)],
    )

    assert outputs == [(True, True, True)]
    assert type(outputs[0]) is tuple


def test_runner_rejects_unsupported_protocol_container_types() -> None:
    import vlabs_integrity

    identity = "def solve(value):\n    return value\n"
    with pytest.raises(TypeError, match="unsupported protocol value"):
        vlabs_integrity._run_source(identity, "solve", [({1, 2},)])
    with pytest.raises(TypeError, match="dictionary keys"):
        vlabs_integrity._run_source(identity, "solve", [({1: "x"},)])


def test_mutation_infrastructure_failure_abstains_instead_of_counting_as_kill(monkeypatch) -> None:
    import vlabs_integrity

    original = vlabs_integrity._run_source

    def unavailable(source, entry, inputs, **kwargs):
        if source == REF:
            return original(source, entry, inputs, **kwargs)
        raise vlabs_integrity.ExecutionInfrastructureError("worker_timeout")

    monkeypatch.setattr(vlabs_integrity, "_run_source", unavailable)
    report = audit_reward(REF, "solve", [([1, 2, 3],)], _gen, n_isomorphic=2)

    assert report.auditable is False
    assert report.failure_reason == "mutation_infrastructure_failure"
    assert report.deployed_kill_rate == 0.0
    assert report.trusted_kill_rate == 0.0
    assert report.gameable is True


def test_candidate_mutant_exception_is_a_real_kill(monkeypatch) -> None:
    import vlabs_integrity

    monkeypatch.setattr(
        vlabs_integrity,
        "_run_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            vlabs_integrity.CandidateExecutionError("ZeroDivisionError")
        ),
    )
    outcome = vlabs_integrity._kills(
        "def solve(value): return 1 / 0",
        "solve",
        [((1,), 1)],
    )
    assert outcome is vlabs_integrity.MutationOutcome.KILLED


def test_reference_worker_failure_is_inconclusive(monkeypatch) -> None:
    import vlabs_integrity

    monkeypatch.setattr(
        vlabs_integrity,
        "_run_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            vlabs_integrity.ExecutionInfrastructureError("spawn_failed")
        ),
    )
    report = audit_reward(REF, "solve", [([1, 2, 3],)], _gen, n_isomorphic=2)
    assert report.auditable is False
    assert report.failure_reason == "reference_infrastructure_failure"
    assert report.gameable is True


def test_child_limits_are_applied_inside_worker_without_preexec(monkeypatch) -> None:
    import vlabs_integrity

    seen = {}

    class FakeProcess:
        pid = 123
        returncode = 0

        def communicate(self, input, timeout):
            seen["input"] = input

        def poll(self):
            return 0

    def fake_popen(*args, **kwargs):
        seen["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(vlabs_integrity.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vlabs_integrity, "_kill_group", lambda _proc: None)
    with pytest.raises(vlabs_integrity.ExecutionInfrastructureError):
        vlabs_integrity._run_source("def solve(): return 1", "solve", [()])

    assert "preexec_fn" not in seen["kwargs"]
    assert "RLIMIT_AS" in vlabs_integrity._RUNNER
    assert "RLIMIT_NPROC" in vlabs_integrity._RUNNER
