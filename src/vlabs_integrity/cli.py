"""`vlabs-integrity` CLI — audit an embedded coding reward (demo), or your own via the API."""
from __future__ import annotations

import json
import sys

from . import audit_reward

_REFERENCE = (
    "def solve(a):\n"
    "    n = 0\n"
    "    for x in a:\n"
    "        if x > 0:\n"
    "            n = n + 1\n"
    "    return n\n"
)


def _demo() -> int:
    def gen(i: int):
        return ([i % 5, (i * 3) % 7, (i * 2) % 4],)

    report = audit_reward(
        reference_source=_REFERENCE,
        entry_point="solve",
        provided_inputs=[([1, 2, 3],)],  # one all-positive test -> misses the >/>= boundary
        input_generator=gen,
    )
    print("vlabs-integrity — gameability audit (judge-free · 0 model calls · $0)\n")
    print(json.dumps(report.to_dict(), indent=2))
    if report.exploitable_gap > 0:
        headline = (
            f"the provided reward MISSES {report.exploitable_gap:.0%} of the exploitable surface "
            f"the isomorphic re-grade catches"
        )
    elif report.gameable:
        headline = "the provided reward is CERTIFIED GAMEABLE (low mutation-kill rate)"
    else:
        headline = "no exploitable gap found on this task"
    print(
        f"\nVerdict: {headline} "
        f"(deployed kill {report.deployed_kill_rate:.0%} vs trusted {report.trusted_kill_rate:.0%})."
    )
    print("Audit your own reward: https://verifiable-labs.com")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in ("demo", "-h", "--help"):
        print("usage: vlabs-integrity [demo]", file=sys.stderr)
        return 2
    if argv and argv[0] in ("-h", "--help"):
        print("vlabs-integrity demo   # audit an embedded coding reward for gameability")
        return 0
    return _demo()


if __name__ == "__main__":
    raise SystemExit(main())
