# vlabs-integrity

**How gameable is your reward? A judge-free, deterministic, 0-model-call audit with no marginal model-provider fee for RL/RFT coding rewards and verifiers.**

```bash
pip install vlabs-integrity
vlabs-integrity demo
```

For teams training with verifiable rewards (RLVR/RFT) and RL-environment authors (e.g. the
[Prime Intellect Environments Hub](https://github.com/PrimeIntellect-ai/verifiers)). Before you
spend GPU-hours training on a coding reward, measure how much of its acceptance surface a policy
can exploit **without solving the task** — no LLM judge, no model calls, no cost.

## What it measures

1. **Mutation-kill certificate** — generate semantics-altering mutants of your trusted reference
   and measure the fraction the deployed verifier **catches**. A low kill rate is the exploitable
   surface an RL policy will find.
2. **Isomorphic shortcut (IPT)** — does a candidate pass the *provided* tests but fail a
   semantics-preserving relabeling (fresh inputs, expected recomputed from the reference)? That is
   a memorizing reward-hack.

```python
from vlabs_integrity import audit_reward

report = audit_reward(
    reference_source="def solve(a):\n    return sum(a)\n",
    entry_point="solve",
    provided_inputs=[([1, 2, 3],)],          # your single/public test(s)
    input_generator=lambda i: ([i % 5, (i*3) % 7],),  # fresh inputs from the task's spec
)
print(report.to_dict())   # deployed_kill_rate, trusted_kill_rate, exploitable_gap, gameable
```

If no semantics-altering mutants can be generated, the report is explicitly
`auditable=false`, `failure_reason="no_semantics_altering_mutants"`, and fails closed with an
integrity reward of `0.0`; absence of mutation evidence is never reported as perfect verifier strength.
Likewise, if the generator cannot supply the requested number of distinct fresh inputs, the report
uses `failure_reason="insufficient_fresh_isomorphic_inputs"` and fails closed. Reports expose both
`requested_isomorphic` and `achieved_isomorphic` so coverage cannot be silently overstated.

Reference and mutant source execute only in scrubbed, ephemeral child processes with time/resource
limits. That subprocess is defense in depth, not a security boundary. Production deployments must
place it inside a disposable, zero-secret, egress-blocked microVM/container with a minimal read-only
filesystem and dedicated identity; local embedded mode is non-promotable for untrusted source.

## verifiers integration (optional)

Add reward-integrity as an auxiliary signal in your `verifiers` rubric:

```python
from vlabs_integrity.verifiers_adapter import integrity_reward_fn
# fn(prompt, completion, answer) -> bounded integrity signal (deterministic, ignores the rollout)
```

The adapter returns `0.0` for every inconclusive or `gameable=true` report. Gap penalties must be
finite and non-negative; callers cannot turn a weak/gameable verifier into a positive reward by
choosing a special penalty value.

`build_integrity_rubric(...)` wraps it in a `verifiers.Rubric` when `pip install "vlabs-integrity[verifiers]"`.

## Honest scope

- **IPT is a public method** ([Helff et al., arXiv:2604.15149](https://arxiv.org/abs/2604.15149)).
  Verifiable Labs **productizes** it — we did not invent it.
- This is a **precondition audit**: a high kill rate bounds the *known* deviation classes a
  verifier catches; it does **not** prove the verifier unhackable.
- Requires a **trusted reference + input generator** (where you can recompute the correct answer
  on fresh inputs). It is not a universal free-form grader.

Self-contained (stdlib only), deterministic, 0 model calls, Apache-2.0. Docs: <https://verifiable-labs.com>
