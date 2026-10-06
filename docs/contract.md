# hhmarl_jax interface contract

Status: **v1 (frozen)**. This is the shared interface for the four parallel agents (A env, B train, C tests, D notebooks/eval/viz). Agents code against it. **Nobody edits this file or a CONTRACT-FROZEN module unilaterally.** If you need a change, put it in your report as a proposal and the orchestrator decides. Inside your own files you are free, provided that the pinned signatures, shapes and field names stay as written here.

References: `docs/spec_original_env.md` ("spec §x") is the behavioural ground truth. `docs/paper_results.md` ("targets") gives the figures to reproduce.

---

## 1. Layout and ownership

| Path | Owner | Content |
|---|---|---|
| `docs/contract.md`, `hhmarl_jax/__init__.py`, `hhmarl_jax/constants.py`, `hhmarl_jax/config.py`, `hhmarl_jax/types.py`, `pyproject.toml` | **orchestrator** (CONTRACT-FROZEN) | dims and constants, config dataclasses, public pytrees, deps |
| `hhmarl_jax/env/` (`state.py`, `geometry.py`, `sim.py`, `scripted.py`, `obs.py`, `rewards.py`, `lowlevel.py`, `hier.py`, `__init__.py`) | **A (env)** | low-level and commander envs |
| `hhmarl_jax/nets/` (`layers.py`, `distributions.py`, `lowlevel.py`, `commander.py`) | **B (train)** | networks and distributions |
| `hhmarl_jax/train/` (`critic.py`, `rollout.py`, `sequences.py`, `ppo.py`, `selfplay.py`, `curriculum.py`, `commander.py`, `checkpoint.py`, `metrics.py`) | **B (train)** | PPO, curriculum, FSP, commander training, checkpoints, logging |
| `tests/` (all files, including `tests/reference/` harness for the original env and `conftest.py`) | **C (tests)** | parity and unit tests |
| `hhmarl_jax/eval/` (`evaluate.py`, `record.py`), `hhmarl_jax/viz/` (`plots.py`, `render.py`), `notebooks/` | **D (notebooks/eval/viz)** | evaluation API, episode records, figures, Kaggle notebooks |

Ownership rules:
- A may **append** private fields to `LowLevelState`/`HierState` and add private helpers. A may not rename or remove pinned fields.
- `state.py` pinned fields (§4.2) are read by C and D.
- C may not edit library code. Bugs found by C go into its report, and the failing test stays in the suite marked `xfail(strict=True)` with a reason.
- D depends only on the public APIs in §4–§6 and §9–§11.
- Read-only for everyone: `envs/`, `warsim/`, `models/`, `config.py`, `train_*.py`, `evaluation.py`.

Dependencies are minimal: `jax`, `flax` (**linen**), `optax`, `numpy`, `msgpack`. Extras: `[viz]` matplotlib and pillow; `[test]` pytest, geographiclib, scipy, gymnasium. There is no orbax, no ray and no torch in the library. On Kaggle, the image's CUDA jax must be kept: notebooks run `pip install -q flax optax msgpack` (plus `-e .` with `--no-deps`) and must **not** `pip install jax` over the preinstalled GPU build.

## 2. Conventions

- Python ≥ 3.10. Everything traced is `jnp.float32` / `jnp.int32` / `bool`. There is no float64 in the library (C may use float64 in reference computations). There is no Python control flow on traced values. Data-dependent loops are `lax.scan`/`fori_loop` over a fixed bound with masks.
- **Static vs dynamic.** `EnvConfig`, `NetConfig`, `PPOConfig`, `RLlibQuirks`, `TrainConfig` and `EvalConfig` are frozen, hashable dataclasses. They are closed over or passed as static args, and changing them recompiles. Per-call values (team sizes, frozen policy params, ratios) are pytrees (`LowLevelEnvParams`, `HierEnvParams`).
- **Slots.** `A = N + M` aircraft slots, with `N = n_agents_max` and `M = n_opps_max`. Slots `0..N-1` are agents (original ids `1..N`) and slots `N..A-1` are opponents. Slot order = original id order = tick update order. Team sizes per episode are `num_agents ≤ N` and `num_opps ≤ M` (dynamic). Unused slots have `exists=False`: they never act, are never hit and are never observed. Original "id 2" = slot 1.
- **Rockets.** There is one rocket slot per aircraft slot (slot k = the rocket of shooter k). Rockets update in `launch_id` order.
- **Positions** are degree offsets from the map origin: `lat − 5.0`, `lon − 7.0`. Flat-earth WGS84 kinematics (spec §1 [PORT]); parity uses tolerances. Headings are compass degrees and speeds are knots. One sim tick = 1 s.
- **RNG.** Every function that needs randomness takes an explicit `key`. Inside a step, keys are split once per draw site. Sim and scripted randomness is drawn up front into explicit draw pytrees (§12). Distributional parity is the goal, not bitwise parity.
- **Batching.** The env functions are written for ONE env. Trainers use `jax.vmap(env.step, in_axes=(0, 0, 0, None))`, with params unbatched. Batched `num_agents`/`num_opps` must also work, so mixed team sizes can be vmapped in one evaluation batch.

## 3. Configuration (`hhmarl_jax/config.py`, `hhmarl_jax/constants.py`)

### 3.1 Objects and defaults
All defaults equal the **original code** values. The code is the source of truth for these values; the summary below is for orientation only.

- `AircraftConfig`: `AC1_CODE` = turn 5 °/s, speed ≤ 900 kn, accel 35, cannon 2.0 km, cone 10°, burst 5 s, p_hit 0.75 (0.15 per tick), 200 s ammo, 5 missiles. `AC2_CODE` = 3.5, 600, 28, 4.5 km, 7°, 3 s, 0.90, 200 s, 0 missiles.
- `EnvConfig`:
  - `kind` lowlevel|hier, `agent_mode`, `level`, `n_agents_max`/`n_opps_max`, `map_size` (0.3 / 0.5 deg);
  - horizon: `horizons_ll=(150,200,300,350,400)`, `horizon_hl=500`;
  - ammo overrides: opponents at L1–4 get (400 s, 8); all units at L5 get (300, 6); HL gets (300, 8);
  - missile cooldowns: LL (7,17), HL (8,12);
  - rewards: `rew_scale=1`, `glob_frac=0`, `friendly_kill=True`, `friendly_punish=False`, killed −2, friendly kill −2, OOB −5 (LL) / −2 (HL), escape fire penalty −0.1, `escape_step_reward="none"`;
  - commander: `n_opp_hl=2`, `hl_max_substeps=16`, situation check when `s > 10`;
  - `quirks: EnvQuirks`, and `record_ticks`.
  - Constructors: `EnvConfig.lowlevel(level, mode)`, `EnvConfig.hier(n, m)`, `EnvConfig.paper(base)`.
- `NetConfig`: `ll_arch` sa|fc, `cmd_arch` gru|sa|fc, `sa_mode` rllib|single_step, `residual_norm` l2|layernorm, `critic_sizes=(175,175,150)`, `extra_fc500=False`, `fc_hidden=500`, `shared_layer=True`, `mha_heads=2`, `max_seq_len=20`.
- `PPOConfig` (low level):
  - lr 1e-4 (Adam .9/.999/1e-8), γ 0.99, λ 0.95, clip 0.25;
  - kl_coeff 0.2 with target 0.025 (adaptive), vf_coeff 1.0, vf_clip 10.0 on the squared error, entropy 0;
  - 30 epochs, minibatch 256 rows per policy, `train_batch_size` 2000 env steps, no grad clip.
  - `PPOConfig.commander()` changes λ to 1.0, kl_target to 0.05 and the batch to 1000.
- `TrainConfig`: `num_envs=100`, `rollout_len=20` (100·20 = 2000 env steps per iteration = the RLlib batch), `iterations=10000` per phase, `ckpt_every=50`, `reward_window=100`, `rllib_faithful=True`, `quirks=None` (resolved via `resolved_quirks()`), plus nested env, net, ppo and curriculum configs.
- `CurriculumConfig`: `phases = L1_fight, L2_fight, L3_fight, L4_fight, L3_escape, L5_fight, commander`, `warm_start=True` (full train state, including optimizer and kl_coeff, carried from L(n−1) fight), `export_every=50`, `fsp_pool_size=1` (latest snapshot only), `escape_l5=False`.
- `EvalConfig`: `kind` commander|fight|escape, `env_kind="hier"` (as in `evaluation.py`: 50 km map, horizon 500), `n_agents`, `n_opps`, `n_episodes=1000`, `num_envs=1000`, `agent_level=5`, `opp_level=4` (commander evaluation forces 5), `opp_fight_ratio=0.75`, `commander_deterministic=True`, `gru_reset_each_step=True`.

### 3.2 Faithfulness switches
- `EnvQuirks` (all True by default) covers the gameplay quirks:
  - sequential id-ordered tick; dead-in-tick units still act;
  - slot-1 rocket fuse; rocket hit check before the move; unguided multiplicative heading noise; no wrap of rocket heading;
  - offset radar cone `[h, h+120]`; cooldown set on a failed launch;
  - one missile attempt for scripted opponents; shared L3 escape timer evaluated per opponent; FSP opponents see the agents' current-step shot flags;
  - heading normalised by /359; rewards of dead killers dropped;
  - HL: frozen opponent distances and order per commander step; escape missile aimed at the farthest opponent; opponents start at 0 kn; P=0.75 of a non-nearest target.
  - A must implement every True branch. False branches are optional: if one is unimplemented, the env constructor raises `NotImplementedError`.
- `RLlibQuirks` (all True = `rllib_faithful`) covers the training-side behaviour:
  - non-causal MHA over zero-padded ≤20-step chunks including future steps (T=1 at rollout);
  - plain loss mean over padded rows (VF regressed to 0 there, KL to the uniform distribution);
  - GAE values computed with zero action slots in the critic, while training uses the real actions;
  - shared 500×500 layer updated by ac1 SGD then ac2 SGD with separate Adam states;
  - horizon treated as terminal (no bootstrap);
  - dead agents' zero-obs/zero-reward rows included;
  - commander λ = 1.0;
  - adaptive KL from the last-epoch mean KL;
  - per-policy advantage standardisation.
  - `RLlibQuirks.clean()` turns the first seven off:
    - `sa_mode` is then expected to be `single_step`;
    - losses are masked;
    - GAE uses the real actions;
    - one joint Adam;
    - truncation bootstraps with `final_value`;
    - dead rows are masked;
    - commander λ = `ppo.gae_lambda`.
- **Not replicated** (documented deviations): RLlib's `complete_episodes` batch overshoot (we use fixed-length fragments and bootstrap at the fragment cut with `last_value`; that cut is not an episode end), the length-1-episode crash, and OS-entropy RNG.

### 3.3 Paper-vs-code options (default = code)

| Option | Code (default) | Paper |
|---|---|---|
| `ac1/ac2.hit_prob` | 0.75 / 0.90 | 0.70 / 0.85 (`AC1_PAPER`, `AC2_PAPER`) |
| `horizons_ll` | 150,200,300,350,400 | 200,250,300,350,400 |
| `l5_opponents` | uniform {L3 fight, L4 fight, L3 escape} | "L1–L4 random" (`"paper"`, may be NotImplemented) |
| `kill_reward` | remaining-ammo shift [1,2]; rocket [1,1.5] | ATA + used-ammo fraction (`"paper"`) |
| `killed_penalty` | −2 | 0 |
| `oob_penalty_hl` | −2 | −5 |
| `escape_step_reward` | `"none"` (`esc_dist_rew=False`) | `"paper"` ±0.01 at <6 km / >13 km; `"code"` = the code's flag-on variant |
| `hl_situation_reward` | 0.1° (~11 km), ATA<15°, opp ATA>40° | `"paper"`: 5 km, 30°, AA 50° |
| `n_opp_hl` | 2 (obs 34, Discrete 3) | 3 (obs 44 with the code's block layout, Discrete 4) |
| `hl_substep_rule` | ≤16 ticks, exit on kill, situation event after s>10 | `"paper"`: T=10, T_h=40, boundary and 2-opponent events |
| `PPOConfig.gamma` / `clip_eps` | 0.99 / 0.25 | 0.95 / 0.2 |
| `NetConfig.critic_sizes` / `extra_fc500` / `residual_norm` | (175,175,150) / False / l2 | (300,100,100) / True / layernorm |

`EnvConfig.paper(base)` flips all env switches at once. Map size stays in degrees (0.3° ≈ 33 km), as in the code.

## 4. Environment API (agent A)

### 4.1 Construction
```python
env = LowLevelEnv(cfg: EnvConfig, opp_logits_fns=(fight_fn, escape_fn) | None)  # fns required iff level >= 4
env = HierEnv(cfg: EnvConfig, ll_logits_fns=(fight_fn, escape_fn))
params = env.default_params(...)                          # LowLevelEnvParams / HierEnvParams
obs, state = env.reset(key, params)
obs, state, reward, done, info = env.step(key, state, actions, params)      # auto-reset
obs, state, reward, done, info = env.step_env(key, state, actions, params)  # no auto-reset
```
`fight_fn`/`escape_fn` are `LogitsFn`s: `fn(params, obs f32[D], ac_type i32[]) -> logits f32[26]`, built by `nets.lowlevel.make_logits_fn(net_cfg, mode)`. They are static Python callables. Their params travel in the env params pytree (§5). A's own tests may pass dummy functions.

### 4.2 State (pinned fields; `hhmarl_jax/env/state.py`)
- `AircraftState` [A]:
  - kinematics: `lat, lon` (offsets), `heading, speed, new_heading, new_speed`;
  - status: `alive, exists, is_agent, ac_type`;
  - weapons: `cannon_remain, cannon_max, burst, missile_remain, rocket_max, missile_wait, has_missile`.
- `RocketState` [A]: `active, lat, lon, heading, new_heading, speed, fire_tick, target, launch_id`.
- `LowLevelState`: `aircraft, rockets, t, side, opp_to_attack[A], opp_pool_idx, opp_mode, l3_escaping, l3_escape_timer, next_launch_id, num_agents, num_opps, episode: EpisodeStats`.
- `HierState`: `aircraft, rockets, t` (sim ticks), `cmd_t, side, opp_sorted[A,max(N,M)], opp_sorted_dist[A,max(N,M)], cmd_action[A], next_launch_id, num_agents, num_opps, episode`.

### 4.3 Low-level env: shapes per step

| Item | Shape / dtype | Notes |
|---|---|---|
| obs | `f32[N, D]`, D = 26 (fight) / 30 (escape) | AC2 rows use `[:24]` / `[:29]`, tail 0. All-zero for a dead or non-existent agent, or when no opponent is alive. Layout spec §6.1/6.2. |
| actions | `i32[N, 4]` | `(heading 0..12, speed 0..8, cannon 0/1, rocket 0/1)`. For AC2 `a[3]` is ignored. Rows of dead or non-existent agents are ignored. |
| reward | `f32[N]` | Spec §8. It is 0 for agents not alive at action time and for non-existent slots. The death step carries −2/−5. |
| done | `bool[]` | `terminated \| truncated` (episode-level; there are no per-agent dones, spec §9). |
| info | `LowLevelInfo` | `terminated, truncated, done, alive_at_action[N], alive[N], exists[N], ac_type[N], final_obs[N,D], events: StepEvents, episode: EpisodeStats` |

Further rules:
- Opponents are driven inside the env:
  - L1–L3 are scripted (spec §5.1).
  - L4/L5 use FSP with **argmax** actions from `params.opp_pool`, on observations built after the agents fired (quirk).
  - The L5 pool entry is sampled once per episode at reset.
- The friend for observations is the fixed pair 1↔2, 3↔4 (`friend_rule_ll="pair"`). For N > 2, `"nearest"` is allowed.
- Escape mode applies the −0.1 fire penalties. Kill rewards are zero in escape mode.
- `t` counts ticks. `truncated` = `t >= horizon` and not terminated.

### 4.4 Commander env: shapes per commander step

| Item | Shape / dtype | Notes |
|---|---|---|
| obs | `f32[N, O]`, O = `cmd_obs_dim(n_opp_hl)` = 34 | Spec §6.3; zeros if dead. |
| actions | `i32[N]` in `[0, 1+n_opp_hl)` | 0 = escape, k = attack the k-th nearest opponent of the frozen sorted list. |
| reward | `f32[N]` | Action assessment (±0.1, −0.1 for an invalid target) + Σ over ticks (+1 kill, −1 killed, −2 OOB). |
| done | `bool[]` | Horizon (500 ticks) checked only between commander steps (t may reach 515). |
| info | `HierInfo` | As LowLevelInfo, plus `n_ticks` (1..16), `effective_action[A]`, `ticks: TickRecord \| None`. |

Further rules:
- Each commander step runs `lax.scan` over 16 ticks with an `active` mask that implements the code's sub-step rule (exit on any kill or OOB event; situation event checked when `s > 10`).
- Inside each tick, every aircraft gets low-level argmax actions:
  - agents use `params.ll_agent`; opponents use `params.ll_opp`;
  - the fight or escape policy is chosen by the effective commander action;
  - opponents' commander actions are sampled in-env with `params.opp_fight_ratio` and the 0.75 non-nearest rule.
- Low-level observations use the frozen distance and order lists of this commander step (quirk).

### 4.5 Dead agents, horizon, auto-reset
- Dead agents keep their slot. They receive zero obs, reward 0 (after the death step), and their actions are ignored. They keep producing rows until the episode ends (`alive_at_action=False`). Non-existent slots have `exists=False` everywhere.
- `terminated` = all existing agents dead or all existing opponents dead. `truncated` = horizon reached and not terminated. The original treats both as terminal (handled trainer-side by `RLlibQuirks.no_bootstrap_at_horizon`).
- Win/loss/draw follow the code exactly:
  - `win = alive_opps==0 and t<horizon`;
  - `loss = alive_agents==0 and t<horizon`;
  - `draw = t>=horizon and both alive`.
  - A wipe-out at exactly `t==horizon` counts as none of the three.
- **Auto-reset** in `step`: if `done`, the env resets with `jax.random.split(key)[1]`, selected with `jnp.where`/`tree_map`. The returned `obs`/`state` are those of the new episode. `info.final_obs`, `info.episode`, `info.events`, `info.ac_type` and `info.alive*` describe the step just taken. `info.episode` is a running accumulator, valid as an end-of-episode summary where `info.done`.

## 5. Policies passed into envs
- Low-level frozen policies are `LowLevelPolicySet(fight=LowLevelParams, escape=LowLevelParams)`. A whole LowLevelParams tree is passed (critic leaves are unused).
- FSP: `OpponentPool(fight=<stacked K_F>, escape=<stacked K_E>, probs=f32[K_F+K_E])`. The pool is unbatched across envs, and the env computes logits for **every** entry and selects by `opp_pool_idx`. This avoids per-env param gathers. Keep K ≤ 4.
  - Code defaults:
    - L4: fight=[L3_fight], probs [1, 0] with a dummy escape entry;
    - L5 fight: fight=[L3_fight, L4_fight], escape=[L3_escape], probs uniform;
    - L5 escape (optional): fight=[L5_fight].
  - L1–L3 use a dummy pool.
- Commander env: `HierEnvParams(ll_agent, ll_opp, opp_fight_ratio, num_agents, num_opps)`.
  - Training: `ll_agent = ll_opp = {L5_fight, L5_escape if present else L3_escape}`.
  - Fight evaluation: `ll_opp.fight = L4_fight`.

## 6. Networks (agent B, flax.linen)

### 6.1 Param pytrees
- `LowLevelParams = {"shared": {"kernel": f32[500,500], "bias": f32[500]}, "ac1": {"actor":…, "critic":…}, "ac2": {"actor":…, "critic":…}}`. There is one tree per mode (fight or escape) and run. The single `"shared"` leaf implements the global `SHARED_LAYER`, shared across types and actor/critic.
- `CommanderParams = {"shared": …, "actor": …, "critic": …}`.
- Inner leaf names are B's choice and must be stable once checkpoints exist.
- Inits match torch: SlimFC orthogonal gain 1 with bias 0; MHA xavier in_proj, torch-Linear out_proj; GRU U(±1/√H).
- Architectures are listed in the `nets/lowlevel.py` and `nets/commander.py` docstrings (spec §10.5). The FC baseline is `in → 500 tanh → SHARED → out` (paper "2×500 tanh").

### 6.2 Low-level apply
```python
init_lowlevel(key, net_cfg, mode) -> LowLevelParams
actor_apply(params, net_cfg, mode, ac_type:int(static), obs f32[B,T,OBS[ac_type]], seq_mask bool[B,T]) -> logits f32[B,T,24|26]
critic_apply(params, net_cfg, mode, ac_type, critic_in f32[B,T,C], seq_mask) -> V f32[B,T]
policy_logits_padded(params, net_cfg, mode, obs f32[...,D], ac_type i32[...]) -> f32[...,26]   # T=1 rollout helper
value_padded(params, net_cfg, mode, critic_in f32[...,C], ac_type i32[...]) -> f32[...]
make_logits_fn(net_cfg, mode) -> LogitsFn                                                     # for envs
```
- In `sa_mode="rllib"`, attention covers the full T axis with no mask.
- At rollout, T=1, so attention reduces to `out_proj(v_proj(x))`.

### 6.3 Commander apply
```python
init_commander(key, net_cfg, n_opp_hl=2) -> CommanderParams
commander_actor_apply(params, net_cfg, obs f32[B,T,O], h0 f32[B,200], seq_mask) -> (logits f32[B,T,n_act], hT)
commander_critic_apply(params, net_cfg, critic_in f32[B,T,3(O+1)], h0, seq_mask) -> (V f32[B,T], hT)
```
- The actor and the critic each carry their own hidden state.
- For the sa and fc archs, `h` is ignored, and `h0` may be passed as `f32[B,200]` zeros.

### 6.4 Distributions (`nets/distributions.py`)
- MultiCategorical over (13, 9, 2, 2) on `logits[...,26]`, with an `ac_type` argument. For AC2 the 4th component is masked: `a[3]=0` and it contributes 0 to logp, entropy and KL.
- Functions: `multicat_{sample,mode,logp,entropy,kl}`, `pad_ac2_logits`, `categorical_{sample,logp,entropy,kl}`.
- FSP and the env use `multicat_mode` (argmax).

## 7. Rollout → PPO data format (agent B)

### 7.1 `Transition` (in `types.py`), time-major `[T, B, N, …]`
- Fields: `obs, critic_obs` (zero action slots), `action, logits` (behaviour), `logp, value` (rollout-time, zero-action critic if faithful), `reward`, and `done/terminated/truncated [T,B]`.
- Masks and indices: `alive` (= `alive_at_action`), `exists, ac_type, ep_step [T,B]`.
- Values and recurrent state: `final_value` (V of `final_obs`, used only by the clean variant), and `h_act, h_val [T,B,N,H]` (state fed *into* the step; H=0 for non-recurrent archs).
- `rollout_fn` also returns `last_value f32[B,N]` for the fragment-cut bootstrap.

### 7.2 Centralised critic input (`train/critic.py`)
- **Low level** (2-agent team, partner = other slot):
  - row layout `[act_own | act_other | obs_own | obs_other]` with unpadded per-type widths, so C = 57 (fight) / 66 (escape) for both types;
  - actions are scaled `(a0/12, a1/8, a2, a3)`, with the AC2 action being 3-wide.
- **Commander** (3 agents): `[a_own, a_lo, a_hi, obs_own, obs_lo, obs_hi]`, with `a/(n_actions−1)`, so C = 105.
- In faithful mode, `value` (used for GAE) comes from zero-action inputs, and the loss recomputes V on inputs with the real actions.

### 7.3 Sequences
- `train/sequences.chunk` cuts each `(env, agent)` stream at episode ends, fragment ends and `ep_step % 20 == 0`. It pads chunks to 20 and stores `h_act0/h_val0` at the chunk start.
- Fight-SA and commander minibatches are drawn from whole chunks (about 256 rows). Escape and FC minibatches are drawn from shuffled rows.

## 8. Training (agent B)

### 8.1 PPO
- `TrainState(params, opt_state{policy: optax}, kl_coeff{policy: f32}, iteration, env_steps, key)`.
- Low-level policies are `"ac1"` and `"ac2"`. With `shared_layer_sequential_adam`, each policy's optimizer covers its own subtree plus `"shared"`, updated ac1 (all epochs) then ac2. Otherwise a single `"joint"` optimizer is used.
- The commander policy is `"commander"`.
- `update(train_state, traj, last_value) -> (train_state, metrics)`. Loss as spec §10.2:
  - `−min(Aρ, A·clip(ρ))` plus `vf_coeff·clip((V−Vt)², 0, vf_clip)` plus `kl_coeff·KL`, with entropy coefficient 0;
  - the reduction follows `loss_mean_over_padding` / `dead_agent_rows_in_loss`.
- Agent steps: low-level 2v2 → policy ac1 = slot 0, ac2 = slot 1 (deterministic in 2v2).

### 8.2 Curriculum (`train/curriculum.py`)
- Phases run in `CurriculumConfig.phases` order; the run dir is `{runs_root}/{exp_name}/{phase}`.
- Fight L(n) restores the latest checkpoint of L(n−1) fight when `warm_start`. L3 escape starts from scratch. L5 escape (optional) restores L3 escape.
- From level ≥ 3, every `export_every` iterations and at the end, the current params are exported to `{runs_root}/{exp_name}/policies/L{level}_{mode}.msgpack` (overwrite = latest). If `fsp_pool_size > 1`, iteration-tagged copies `…_it{iter:07d}.msgpack` are also kept.
- `env_steps` is per phase. `env_steps_total` accumulates across warm-started phases.

### 8.3 FSP
- `selfplay.build_opponent_pool(policies_dir, level, …)` builds the OpponentPool from the files above, per §5.
- The pool is fixed for a whole phase (faithful: loaded once at env construction).

### 8.4 Commander (`train/commander.py`)
- Setup: HierEnv 3v3 with frozen LL params from `policies/`, `PPOConfig.commander()`, `cmd_arch` gru|sa|fc.
- `env_steps` = commander steps (the x-axis of Fig 10a). Sim ticks are logged separately.

## 9. Metrics and run directory (B writes, D reads)

```
runs/{exp_name}/
  {phase}/                     phase ∈ L1_fight..L5_fight, L3_escape, L5_escape, commander
    config.json                dataclasses.asdict(TrainConfig) + {"phase", "git_rev", "jax_version", "device"}
    metrics.jsonl              one JSON object per iteration (below)
    ckpt/iter_{k:07d}.msgpack  + iter_{k:07d}.json (meta) + latest.json {"path", "iteration"}
  policies/L{level}_{fight|escape}.msgpack, commander.msgpack (+ optional _it{k:07d} copies)
  eval/{kind}_{n}v{m}.json     EvalResult JSON (§11)
  episodes/{kind}_{n}v{m}_ep{k}.npz   EpisodeRecord (§11.3)
  figures/                     notebook outputs (Kaggle: also copied to /kaggle/working/figures)
```

`metrics.jsonl` record. Required keys:
- Identity and step counters: `iter`, `phase`, `level`, `mode`, `arch`, `env_steps` (cumulative this phase; commander = commander steps), `env_steps_total`, `sim_ticks`, `agent_steps`.
- Episode statistics:
  - `episodes` (completed this iteration);
  - `episode_reward_mean`: RLlib-style, the mean team-summed return over the last `reward_window=100` completed episodes (this is the y-axis of Figs 6, 8a and 10a);
  - `episode_reward_mean_iter`: episodes of this iteration only, NaN if none;
  - `episode_len_mean`, `win_rate`, `loss_rate`, `draw_rate` (window 100);
  - `kills_ac1`, `kills_ac2`, `killed_ac1`, `killed_ac2`, `fk_ac1`, `fk_ac2` (sums this iteration).
- Per-policy losses: `pi/{policy}/{pi_loss, vf_loss, kl, entropy, kl_coeff, clip_frac, explained_var}`.
- Timing: `time/sample_s`, `time/train_s`, `time/iter_s`, `sps` (env steps per second).

Unknown extra keys are allowed. Values must be JSON numbers; NaN is written as `NaN` (Python `json` default). Readers use `train.metrics.read_metrics(run_dir)`.

## 10. Checkpoints
- Train checkpoint: `flax.serialization.msgpack_serialize(to_state_dict({"train_state": TrainState with key as jax.random.key_data}))`. A sidecar JSON holds `{"iteration", "env_steps", "env_steps_total", "phase", "net_cfg", "ppo_cfg", "env_cfg", "rllib_faithful"}`. Restoring needs a template built from `init_*` and the optimizer.
- Policy file: msgpack `{"params": state_dict(params), "meta": json_string}`, with meta `{"kind": "lowlevel"|"commander", "mode", "level", "net_cfg": asdict(NetConfig), "n_opp_hl", "iteration", "env_steps"}`. `load_policy(path) -> (params, meta)` rebuilds the template from meta, so D can load policies without a TrainState.

## 11. Evaluation (agent D)

### 11.1 API (`hhmarl_jax/eval/evaluate.py`)
```python
evaluate(cfg: EvalConfig, agent_ll: LowLevelPolicySet, opp_ll: LowLevelPolicySet,
         commander_params=None, ll_net_cfg=NetConfig(), cmd_net_cfg=NetConfig()) -> EvalResult
eval_sweep(cfg, sizes=((2,2),(3,3),(4,4),(5,5)), ...) -> {"2v2": EvalResult, ...}
save_eval(result, path) / load_eval(path)
```
- Faithful defaults run in HierEnv (map 0.5, horizon 500) as `evaluation.py` does.
- The three kinds:
  - `"commander"`: agents use the commander (argmax, GRU quirk), and both teams use L5;
  - `"fight"` (Fig 7a/7b): every agent's action is 1 (nearest), and opponents use `ll_opp` = L4 fight, with `opp_fight_ratio` 0.75 (code) or 1.0 (pure fight);
  - `"escape"` (Fig 8b, no original script): every agent's action is 0, and opponents use L5 fight with `opp_fight_ratio=1.0` by default.
- `env_kind="lowlevel"` (30 km, 2v2, LL horizon) is optional.
- Exactly `n_episodes` episodes are counted, from the first finished episodes of each env, in a deterministic order.

### 11.2 `EvalResult` and its JSON
Fields: `n_episodes, win, loss, draw, kills (k-1, k-2), killed_by_opp (d-1, d-2), oob, fk_by_killer (fk-1, fk-2), fk_by_victim, escaped, killed, kills_cat, mean_return, mean_length, cmd_counts, config`. The JSON is `result._asdict()`, with tuples as lists and rates added as `win_rate`, `loss_rate` and `draw_rate`.

Definitions:
- Fig 7a:
  - k-t = opponents destroyed by agents of type t (cannon or rocket);
  - d-t = agents of type t shot down by opponents (OOB is reported separately in `oob`);
  - fk-t = friendly kills committed by type t.
- Fig 8b categories are mutually exclusive, checked in this order:
  - `killed`: ≥1 agent lost, counting shot down + OOB + friendly;
  - `kills_cat`: else ≥1 opponent destroyed;
  - `escaped`: otherwise.

### 11.3 Episode records
`record.record_episode(...) -> EpisodeRecord` holds numpy arrays at sim-tick resolution, as defined in `eval/record.py` (absolute lat/lon). For HierEnv they come from `info.ticks` with `EnvConfig.record_ticks=True`; for LowLevelEnv they come from per-step `state.aircraft/rockets`. Saved with `save_record` as compressed npz, with meta as a JSON string.

## 12. Test hooks (A provides, C uses)
- `env/geometry.py` pure functions (signatures pinned in the file): `direct, distance_km, bearing_deg, signed_heading_diff, planar_distance, focus_angle, heading_diff_norm`.
- `env/sim.py`: `tick(aircraft, rockets, t, draws: TickDraws, cfg) -> (aircraft, rockets, TickEvents)` and `try_fire_missile(...)`. All tick randomness is in `TickDraws(cannon_u f32[A,A], rocket_noise f32[A])`, so C can monkeypatch the original `rnd_gen.random`/`random.uniform` to the same values.
- `env/scripted.py`: `ScriptDraws` (A documents its field layout in the docstring), `scripted_opponent_step(state, draws, cfg)`.
- `env/obs.py`: `fight_obs`, `escape_obs`, `commander_obs`, `nearest_enemies`.
- C builds states by constructing `LowLevelState` directly from pinned fields. A keeps appended private fields constructible with defaults via a helper `state.py: make_lowlevel_state(**pinned) / make_hier_state(**pinned)`, which A adds.
- Tolerances: positions 1e-6 deg (about 0.1 m) per tick; angles 0.05°; obs 1e-4.
- Statistical tests (hit rates, scripted behaviour, outcome rates) use ≥ 10k samples and binomial 4σ bounds.
- The original env imports ray, torch and cairo/cartopy. C's harness stubs those modules via `sys.modules` before importing `envs/`.

## 13. Notebooks (agent D)
- Notebooks live in `notebooks/` and are Kaggle-ready: a pip cell first, then repo clone/install, then GPU check (`jax.devices()`).
- `01_train_lowlevel.ipynb` covers the curriculum (Fig 6 runs `sa_curr`, `sa_nocurr` = L3_fight from scratch, `fc_curr`) and escape. `02_train_commander.ipynb` covers the gru, sa and fc commanders. `03_evaluate_and_plot.ipynb` covers Figs 6, 7a, 7b, 8a, 8b, 10a, 10b and renders/animations.
- Budgets are configurable, with a short smoke-run preset. Figures are saved to `/kaggle/working/figures`.
- D only consumes files in §9–§11 and public APIs, so its notebooks work on synthetic `metrics.jsonl` before training exists.

## 14. Change log
- v1: initial contract.

## Orchestrator decisions (binding)
1. **Presets.** Two training presets: `faithful` (default) keeps the original env-steps per PPO update, minibatch size, epochs and max_seq_len, so learning per env-step matches RLlib; `throughput` uses many more parallel envs (e.g. 1024+) with a proportionally larger batch. Paper-reproduction runs use `faithful`; `throughput` is for exploration and must be labelled as such in logs.
2. **Fig 7b eval fight ratio.** Follow the original `evaluation.py` behaviour by default; expose the fight/escape ratio as an eval argument (1.0 = pure fight).
3. **Fig 8b escape eval.** Accepted as proposed (L5 fight opponents that always fight; outcome precedence killed/OOB=lost > kills > escaped). Document it as our own protocol, since the original has no script.
4. **Clean SA variant.** Single-step attention is sufficient.
5. **Rollout cut.** Fixed-length rollouts with value bootstrap at the cut are accepted (RLlib `truncate_episodes` also bootstraps at fragment cuts); horizon end stays non-bootstrapped in `rllib_faithful` mode.
6. **"Paper" options** (paper L5 opponent mix, paper commander sub-step/event rule) are out of scope for now; `NotImplementedError` is fine.
