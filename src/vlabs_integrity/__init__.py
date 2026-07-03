"""vlabs-integrity — how gameable is your reward/verifier? A judge-free, deterministic, $0 audit.

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
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

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


def _fn_from_source(src: str, entry: str) -> Callable:
    ns: dict = {}
    exec(compile(src, "<candidate>", "exec"), ns)  # noqa: S102 - trusted reference / our mutants only
    return ns[entry]


def _kills(mutant_src: str, entry: str, cases: Sequence[tuple]) -> bool:
    """True iff the verifier's cases catch the mutant (a wrong output or a crash)."""
    try:
        fn = _fn_from_source(mutant_src, entry)
    except BaseException:  # noqa: BLE001 - a mutant that won't even load is caught
        return True
    for args, expected in cases:
        try:
            if not _eq(fn(*args), expected):
                return True
        except BaseException:  # noqa: BLE001
            return True
    return False


@dataclass(frozen=True)
class GameabilityReport:
    entry: str
    n_mutants: int
    deployed_kill_rate: float   # fraction of mutants the PROVIDED tests catch
    trusted_kill_rate: float    # fraction the isomorphic re-grade bank catches
    exploitable_gap: float      # trusted - deployed: surface the provided reward misses
    gameable: bool
    isomorphic_shortcut: bool   # does the reference-vs-provided suite already differ?
    honesty: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entry": self.entry, "n_mutants": self.n_mutants,
            "deployed_kill_rate": round(self.deployed_kill_rate, 4),
            "trusted_kill_rate": round(self.trusted_kill_rate, 4),
            "exploitable_gap": round(self.exploitable_gap, 4),
            "gameable": self.gameable, "isomorphic_shortcut": self.isomorphic_shortcut,
            "honesty": self.honesty,
        }


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
    reference = _fn_from_source(reference_source, entry_point)
    provided_cases = [(a, reference(*a)) for a in provided_inputs]

    seen = {repr(a) for a in provided_inputs}
    iso_inputs: list[tuple] = []
    i = 1
    while len(iso_inputs) < n_isomorphic and i <= n_isomorphic * 4:
        a = input_generator(i)
        i += 1
        if repr(a) in seen:
            continue
        seen.add(repr(a))
        iso_inputs.append(a)
    iso_cases = [(a, reference(*a)) for a in iso_inputs]

    mutants = _mutate_source(reference_source, max_mutants=max_mutants)
    if not mutants:
        deployed = trusted = 1.0
    else:
        deployed = sum(_kills(m, entry_point, provided_cases) for m in mutants) / len(mutants)
        trusted = sum(_kills(m, entry_point, provided_cases + iso_cases) for m in mutants) / len(mutants)
    gap = max(0.0, trusted - deployed)
    honesty = [
        "IPT is a public method (Helff et al., arXiv:2604.15149); Verifiable Labs productizes it.",
        "A precondition audit: kill rate bounds KNOWN deviation classes, it does not prove the "
        "verifier unhackable.",
        "Requires a trusted reference + input generator; not a universal free-form grader.",
    ]
    return GameabilityReport(
        entry=entry_point, n_mutants=len(mutants),
        deployed_kill_rate=deployed, trusted_kill_rate=trusted,
        exploitable_gap=gap, gameable=deployed < weak_threshold,
        isomorphic_shortcut=gap > 0.0, honesty=honesty,
    )


__all__ = ["GameabilityReport", "audit_reward", "__version__"]
