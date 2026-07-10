"""vlabs-integrity — a judge-free, deterministic, 0-model-call gameability audit.

For RL/RFT teams and RL-environment authors (e.g. the Prime Intellect Environments Hub). Before
you train on a coding reward, measure how much of its acceptance surface a policy can exploit
WITHOUT solving the task, using two judge-free signals:

  1. Mutation-kill certificate — generate semantics-altering mutants of the trusted reference and
     measure the fraction the deployed verifier CATCHES. A low kill rate is the exploitable
     surface an RL policy will find.
  2. Isomorphic shortcut check (IPT) — does a candidate pass the provided tests but fail a
     semantics-preserving relabeling of them (fresh inputs, expected recomputed from the
     reference)? That is a memorizing reward-hack.

Self-contained (stdlib only: `ast` + `subprocess`), deterministic, 0 model calls. IPT is a public
method (Helff et al., arXiv:2604.15149); Verifiable Labs productizes it. This is a precondition
audit — a high kill rate bounds KNOWN deviation classes a verifier catches, it does not prove the
verifier unhackable.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .codec import WORKER_CODEC_SOURCE, decode_value, encode_value

__version__ = "0.1.0"

_BINOP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
          ast.Div: ast.Mult, ast.FloorDiv: ast.Mult, ast.Mod: ast.Mult}
_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


def _mutate_source(src: str, *, max_mutants: int = 40, seed: int = 0) -> list[str]:
    """Up to ``max_mutants`` distinct semantics-altering source mutants (one op flip each)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    nodes = list(ast.walk(tree))
    ops: list[tuple[int, str]] = []
    for i, n in enumerate(nodes):
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOP:
            ops.append((i, "bin"))
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
            ops.append((i, "cmp"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            ops.append((i, "const"))
    random.Random(seed).shuffle(ops)
    out: list[str] = []
    seen = {src}
    for idx, kind in ops:
        if len(out) >= max_mutants:
            break
        t2 = ast.parse(src)
        node = list(ast.walk(t2))[idx]
        if kind == "bin":
            node.op = _BINOP[type(node.op)]()
        elif kind == "cmp":
            node.ops = [_CMP[type(node.ops[0])]()]
        else:
            node.value = node.value + 1
        ast.fix_missing_locations(t2)
        try:
            m = ast.unparse(t2)
        except Exception:  # noqa: BLE001
            continue
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _eq(a, b) -> bool:
    return a == b


_RUNNER = r"""
import contextlib, json, os, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (1_073_741_824, 1_073_741_824))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (OSError, ValueError):
            pass
except (AttributeError, ImportError, OSError, ValueError):
    pass
""" + WORKER_CODEC_SOURCE + r"""
p = json.loads(sys.stdin.read())
ns = {}
result = {"ok": False, "outputs": [], "error": None}
try:
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(p["source"], "<audit-child>", "exec"), ns)
        fn = ns.get(p["entry"])
        if not callable(fn):
            raise ValueError("entry point is not callable")
        outputs = []
        for encoded_args in p["inputs"]:
            args = _decode_value(encoded_args)
            if type(args) is not tuple:
                raise ValueError("arguments must decode to a tuple")
            value = fn(*args)
            outputs.append(_encode_value(value))
    result = {"ok": True, "outputs": outputs, "error": None}
except BaseException as exc:
    result = {"ok": False, "outputs": [], "error": type(exc).__name__}
sys.stdout.write(json.dumps(result, allow_nan=False))
"""


class ExecutionInfrastructureError(RuntimeError):
    """The isolated worker could not produce a trustworthy protocol response."""


class CandidateExecutionError(RuntimeError):
    """The submitted source ran in the worker but did not produce candidate outputs."""


class MutationOutcome(str, Enum):
    """A verifier trial outcome that cannot confuse infrastructure with a kill."""

    KILLED = "killed"
    SURVIVED = "survived"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def _run_source(src: str, entry: str, inputs: Sequence[tuple], *, timeout_s: float = 5.0) -> list:
    """Run source in a scrubbed, ephemeral child and return JSON-safe outputs."""
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be a finite positive number")
    payload = json.dumps(
        {"source": src, "entry": entry, "inputs": [encode_value(args) for args in inputs]},
        allow_nan=False,
    )
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ") if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    workdir = tempfile.mkdtemp(prefix="vlabs_integrity_")
    proc = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-I", "-c", _RUNNER],
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    cwd=workdir,
                    env=env,
                    text=True,
                    start_new_session=(os.name == "posix"),
                )
            except OSError as exc:
                raise ExecutionInfrastructureError("worker_spawn_failure") from exc
            try:
                proc.communicate(input=payload, timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                _kill_group(proc)
                proc.wait(timeout=2)
                raise ExecutionInfrastructureError("worker_timeout") from exc
            finally:
                # The direct child may have forked descendants; reap the owned session on clean exit too.
                _kill_group(proc)
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(64_001).decode("utf-8", errors="replace")
            stderr_file.read(64_001)
        if proc.returncode != 0:
            raise ExecutionInfrastructureError(
                f"worker_nonzero_exit:{proc.returncode}"
            )
        try:
            result = json.loads(stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ExecutionInfrastructureError("invalid_worker_json") from exc
        if type(result) is not dict or set(result) != {"ok", "outputs", "error"}:
            raise ExecutionInfrastructureError("invalid_worker_schema")
        if type(result["ok"]) is not bool:
            raise ExecutionInfrastructureError("invalid_worker_status")
        if not result["ok"]:
            error_class = result["error"] if type(result["error"]) is str else "CandidateError"
            raise CandidateExecutionError(error_class[:128])
        outputs = result.get("outputs")
        if type(outputs) is not list or len(outputs) != len(inputs):
            raise ExecutionInfrastructureError("invalid_worker_output_count")
        try:
            return [decode_value(output) for output in outputs]
        except (TypeError, ValueError) as exc:
            raise ExecutionInfrastructureError("invalid_worker_tagged_output") from exc
    finally:
        if proc is not None and proc.poll() is None:
            _kill_group(proc)
        shutil.rmtree(workdir, ignore_errors=True)


def _kills(mutant_src: str, entry: str, cases: Sequence[tuple]) -> MutationOutcome:
    """Classify a mutant trial without counting worker failures as verifier strength."""
    try:
        outputs = _run_source(mutant_src, entry, [args for args, _expected in cases])
    except CandidateExecutionError:
        return MutationOutcome.KILLED
    except (ExecutionInfrastructureError, TypeError, ValueError, OSError):
        return MutationOutcome.INFRASTRUCTURE_FAILURE
    for output, (_args, expected) in zip(outputs, cases):
        if not _eq(output, expected):
            return MutationOutcome.KILLED
    if len(outputs) != len(cases):
        return MutationOutcome.INFRASTRUCTURE_FAILURE
    return MutationOutcome.SURVIVED


@dataclass(frozen=True)
class GameabilityReport:
    entry: str
    n_mutants: int
    deployed_kill_rate: float   # fraction of mutants the PROVIDED tests catch
    trusted_kill_rate: float    # fraction the isomorphic re-grade bank catches
    exploitable_gap: float      # trusted - deployed: surface the provided reward misses
    gameable: bool
    isomorphic_shortcut: bool   # does the reference-vs-provided suite already differ?
    requested_isomorphic: int
    achieved_isomorphic: int
    auditable: bool = True
    failure_reason: Optional[str] = None
    honesty: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entry": self.entry, "n_mutants": self.n_mutants,
            "deployed_kill_rate": round(self.deployed_kill_rate, 4),
            "trusted_kill_rate": round(self.trusted_kill_rate, 4),
            "exploitable_gap": round(self.exploitable_gap, 4),
            "gameable": self.gameable, "isomorphic_shortcut": self.isomorphic_shortcut,
            "requested_isomorphic": self.requested_isomorphic,
            "achieved_isomorphic": self.achieved_isomorphic,
            "auditable": self.auditable, "failure_reason": self.failure_reason,
            "honesty": self.honesty,
        }


def _inconclusive_report(
    entry_point: str,
    *,
    requested_isomorphic: int,
    achieved_isomorphic: int,
    failure_reason: str,
    detail: str,
    n_mutants: int = 0,
) -> GameabilityReport:
    """Build the one fail-closed shape used by every non-auditable path."""
    return GameabilityReport(
        entry=entry_point,
        n_mutants=n_mutants,
        deployed_kill_rate=0.0,
        trusted_kill_rate=0.0,
        exploitable_gap=0.0,
        gameable=True,
        isomorphic_shortcut=False,
        requested_isomorphic=requested_isomorphic,
        achieved_isomorphic=achieved_isomorphic,
        auditable=False,
        failure_reason=failure_reason,
        honesty=[
            "IPT is a public method (Helff et al., arXiv:2604.15149); "
            "Verifiable Labs productizes it.",
            "A precondition audit: kill rate bounds KNOWN deviation classes, it does not "
            "prove the verifier unhackable.",
            "Requires a trusted, deterministic reference + input generator; not a universal "
            "free-form grader.",
            f"INCONCLUSIVE: {detail}; result fails closed.",
        ],
    )


def audit_reward(
    reference_source: str,
    entry_point: str,
    provided_inputs: Sequence[tuple],
    input_generator: Callable[[int], tuple],
    *,
    n_isomorphic: int = 16,
    max_mutants: int = 40,
    weak_threshold: float = 0.75,
) -> GameabilityReport:
    """Audit how gameable a coding reward is.

    ``provided_inputs`` and ``input_generator(i)`` return ARGS TUPLES. Expected outputs are always
    recomputed from ``reference_source`` (never a stored key). Deterministic + judge-free.
    """
    if type(n_isomorphic) is not int or n_isomorphic < 1:
        raise ValueError("n_isomorphic must be an integer >= 1")
    if type(max_mutants) is not int or max_mutants < 1:
        raise ValueError("max_mutants must be an integer >= 1")
    if (
        isinstance(weak_threshold, bool)
        or not isinstance(weak_threshold, (int, float))
        or not math.isfinite(weak_threshold)
        or not 0.0 <= weak_threshold <= 1.0
    ):
        raise ValueError("weak_threshold must be a finite number in [0, 1]")
    if not isinstance(reference_source, str) or not reference_source.strip():
        raise ValueError("reference_source must be non-empty source text")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError("entry_point must be a valid identifier")
    if not provided_inputs:
        raise ValueError("provided_inputs must contain at least one case")
    if any(type(args) is not tuple for args in provided_inputs):
        raise ValueError("every provided input must be an argument tuple")

    seen = {repr(a) for a in provided_inputs}
    iso_inputs: list[tuple] = []
    i = 1
    while len(iso_inputs) < n_isomorphic and i <= n_isomorphic * 4:
        try:
            a = input_generator(i)
        except Exception:  # noqa: BLE001 - caller generator failures are an abstention
            return _inconclusive_report(
                entry_point,
                requested_isomorphic=n_isomorphic,
                achieved_isomorphic=len(iso_inputs),
                failure_reason="input_generator_failure",
                detail="the input generator raised before coverage was complete",
            )
        i += 1
        if type(a) is not tuple:
            return _inconclusive_report(
                entry_point,
                requested_isomorphic=n_isomorphic,
                achieved_isomorphic=len(iso_inputs),
                failure_reason="invalid_generated_input",
                detail="the input generator returned a non-tuple argument set",
            )
        if repr(a) in seen:
            continue
        seen.add(repr(a))
        iso_inputs.append(a)
    achieved_isomorphic = len(iso_inputs)
    if achieved_isomorphic != n_isomorphic:
        return _inconclusive_report(
            entry_point,
            requested_isomorphic=n_isomorphic,
            achieved_isomorphic=achieved_isomorphic,
            failure_reason="insufficient_fresh_isomorphic_inputs",
            detail="the input generator did not produce the requested number of distinct fresh inputs",
        )
    all_inputs = list(provided_inputs) + iso_inputs
    try:
        expected = _run_source(reference_source, entry_point, all_inputs)
        repeated_expected = _run_source(reference_source, entry_point, all_inputs)
    except ExecutionInfrastructureError:
        return _inconclusive_report(
            entry_point,
            requested_isomorphic=n_isomorphic,
            achieved_isomorphic=achieved_isomorphic,
            failure_reason="reference_infrastructure_failure",
            detail="the isolated reference worker failed",
        )
    except (CandidateExecutionError, TypeError, ValueError, OSError):
        return _inconclusive_report(
            entry_point,
            requested_isomorphic=n_isomorphic,
            achieved_isomorphic=achieved_isomorphic,
            failure_reason="reference_execution_failure",
            detail="the reference source or its protocol values were invalid",
        )
    if repeated_expected != expected:
        return _inconclusive_report(
            entry_point,
            requested_isomorphic=n_isomorphic,
            achieved_isomorphic=achieved_isomorphic,
            failure_reason="nondeterministic_reference",
            detail="independent reference executions disagreed",
        )
    provided_count = len(provided_inputs)
    provided_cases = list(zip(provided_inputs, expected[:provided_count]))
    iso_cases = list(zip(iso_inputs, expected[provided_count:]))

    mutants = _mutate_source(reference_source, max_mutants=max_mutants)
    if not mutants:
        # No mutation evidence is not perfect integrity. Mark the audit inconclusive and
        # fail closed so gates/adapters cannot reward an untested verifier with 1.0.
        deployed = trusted = 0.0
        auditable = False
        failure_reason = "no_semantics_altering_mutants"
    else:
        deployed_outcomes = [_kills(m, entry_point, provided_cases) for m in mutants]
        trusted_outcomes = [
            _kills(m, entry_point, provided_cases + iso_cases) for m in mutants
        ]
        if MutationOutcome.INFRASTRUCTURE_FAILURE in (
            *deployed_outcomes,
            *trusted_outcomes,
        ):
            return _inconclusive_report(
                entry_point,
                requested_isomorphic=n_isomorphic,
                achieved_isomorphic=achieved_isomorphic,
                failure_reason="mutation_infrastructure_failure",
                detail="at least one mutant trial lacked a trustworthy worker result",
                n_mutants=len(mutants),
            )
        deployed = deployed_outcomes.count(MutationOutcome.KILLED) / len(mutants)
        trusted = trusted_outcomes.count(MutationOutcome.KILLED) / len(mutants)
        auditable = True
        failure_reason = None
    gap = max(0.0, trusted - deployed)
    honesty = [
        "IPT is a public method (Helff et al., arXiv:2604.15149); Verifiable Labs productizes it.",
        "A precondition audit: kill rate bounds KNOWN deviation classes, it does not prove the "
        "verifier unhackable.",
        "Requires a trusted reference + input generator; not a universal free-form grader.",
    ]
    if not auditable:
        honesty.append("INCONCLUSIVE: no semantics-altering mutants were generated; result fails closed.")
    return GameabilityReport(
        entry=entry_point, n_mutants=len(mutants),
        deployed_kill_rate=deployed, trusted_kill_rate=trusted,
        exploitable_gap=gap, gameable=(not auditable or deployed < weak_threshold),
        isomorphic_shortcut=gap > 0.0,
        requested_isomorphic=n_isomorphic,
        achieved_isomorphic=achieved_isomorphic,
        auditable=auditable,
        failure_reason=failure_reason, honesty=honesty,
    )


__all__ = [
    "CandidateExecutionError",
    "ExecutionInfrastructureError",
    "GameabilityReport",
    "MutationOutcome",
    "audit_reward",
    "__version__",
]
