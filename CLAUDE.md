# CLAUDE.md — HHMARL 2D (JAX/GPU reproduction)

## Project layout
- `envs/`, `warsim/`, `config.py`, `train_*.py`, `models/`: ORIGINAL IDSIA/hhmarl_2D code (RLlib, CPU). Treat as the reference implementation; do not edit.
- `hhmarl_jax/`: GPU/JAX re-implementation (vectorized env + training).
- `notebooks/`: Kaggle notebooks (training, evaluation, visualization).
- `tests/`: parity tests (JAX env vs. original env) and unit tests.
- `docs/`: specs and reports. `docs/spec_original_env.md` (original code spec), `docs/paper_results.md` (paper targets), `docs/contract.md` (shared interface contract for hhmarl_jax).

## Orchestration protocol (token efficiency)
The main session is the **orchestrator** ("chef"). It talks to the user, plans, splits work, and reviews; it does not do bulk reading or write large amounts of code itself.

1. **Delegate work to sub-agents.** Bulk reading (papers, large files, logs) and code writing go to sub-agents. The orchestrator only receives their short reports, so its context stays small and it can track the whole project for long.
2. **Loose briefs.** A brief states the goal, the files the agent owns, what it must not touch, and the expected output. Do not micro-specify; give the model room.
3. **Draw ownership boundaries.** Never send two agents to the same file. Each agent owns a disjoint set of files. Shared interfaces live in `docs/contract.md`, written before parallel work starts; agents code against it and must not change it unilaterally (report a proposed change instead).
4. **Contract first, then parallel.** For multi-agent implementation: one agent (or the orchestrator) writes the contract, then agents work in parallel against it. Use git worktrees when agents would touch overlapping areas.
5. **Short reports.** Sub-agents return a summary of at most ~250 words: what changed, where, how it was verified, open risks. Long findings go into files under `docs/`, not into the report.
6. **Shared notes.** When several agents scan the same material, each writes its notes to its own file under `docs/notes/`; a later agent merges them.
7. **Commit often.** Work must not be lost if a session dies. Commit at every meaningful milestone with clear messages. The orchestrator commits after each phase when agents share a working tree.
8. **Real tests.** A passing test must mean something. For a bug fix, add a test that fails without the fix. Use a separate reviewer agent for critical pieces (env dynamics, reward, PPO loss).
9. **Not everything is parallel.** Sequential dependencies run sequentially.
10. **Turn repeated problems into rules.** If the same mistake happens twice, add a rule here.

## Conventions
- Python 3.10+, JAX + Flax/Optax for `hhmarl_jax`. Pure functions, `jax.jit`/`jax.vmap` friendly, no Python control flow on traced values.
- Faithfulness first: when the original code and the paper disagree, follow the original code by default and record the discrepancy in `docs/`.
- Notebooks must run on Kaggle (GPU T4/P100) with `pip install` cells at the top and save figures to `/kaggle/working/figures`.
