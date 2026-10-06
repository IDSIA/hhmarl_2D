"""Configuration objects.  CONTRACT-FROZEN (owner: orchestrator).

All config classes are frozen (hashable) dataclasses so they can be closed over or passed as
`static_argnums` to `jax.jit`.  Defaults follow the ORIGINAL CODE (IDSIA/hhmarl_2D HEAD).
Paper values are available as explicit options (see docs/contract.md §3.3).

Agents may NOT add/rename fields here unilaterally; propose changes to the orchestrator.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

# ======================================================================== aircraft


@dataclass(frozen=True)
class AircraftConfig:
    """Per-type aircraft / weapon constants (spec §2.1)."""
    max_turn_deg: float          # deg per tick
    max_speed: float             # knots; also speed-obs normaliser
    max_accel_kn: float          # knots per tick
    cannon_range_km: float       # strict "<"
    cannon_width_deg: float      # full cone width; hit if |bearing diff| <= width/2
    burst_s: int                 # burst length (ticks)
    hit_prob: float              # per burst; per-tick prob = hit_prob / burst_s
    cannon_ammo_s: float = 200.0  # class default (seconds of burst)
    missiles: int = 0            # class default
    missile_range_km: float = 111.0
    missile_radar_width_deg: float = 120.0


AC1_CODE = AircraftConfig(5.0, 900.0, 35.0, 2.0, 10.0, 5, 0.75, 200.0, 5)
AC2_CODE = AircraftConfig(3.5, 600.0, 28.0, 4.5, 7.0, 3, 0.90, 200.0, 0)
AC1_PAPER = dataclasses.replace(AC1_CODE, hit_prob=0.70)
AC2_PAPER = dataclasses.replace(AC2_CODE, hit_prob=0.85)

# ======================================================================== env quirks


@dataclass(frozen=True)
class EnvQuirks:
    """Original-code gameplay quirks.  True = replicate (default).

    The env owner MUST implement every quirk with value True.  A False ("fixed") branch is
    optional: if not implemented, the env constructor raises NotImplementedError for it.
    """
    sequential_tick: bool = True             # id-ordered per-aircraft update, then rockets by launch id
    dead_units_act_in_tick: bool = True      # unit killed earlier in the tick still shoots/moves
    rocket_fuse_on_slot1: bool = True        # every rocket detonates on aircraft id 2 within 1 km
    rocket_hit_check_before_move: bool = True
    unguided_missiles: bool = True           # heading *= U(0.95,1.05) each tick, no guidance
    rocket_heading_no_wrap: bool = True
    radar_cone_offset: bool = True           # launch only if bearing in [h, h+120] (int-truncated)
    cooldown_on_failed_launch: bool = True
    scripted_single_missile_attempt: bool = True  # scripted opps never decrement missile_wait
    l3_shared_escape_timer: bool = True      # L3 escape flag/timer evaluated per opponent
    fsp_opp_sees_current_shots: bool = True  # FSP opp obs built after agents fired this step
    heading_norm_359: bool = True            # heading_norm = (h % 359) / 359
    drop_rewards_of_dead_killers: bool = True
    hl_frozen_opp_distances: bool = True     # LL obs distances/order frozen per commander step
    hl_escape_missile_targets_farthest: bool = True
    hl_opp_start_speed_zero: bool = True     # args.level == 1 in HL => opponents start at 0 kn
    hl_opp_attacks_non_nearest: bool = True  # P=0.75 to target a random non-nearest agent


# ======================================================================== env


@dataclass(frozen=True)
class EnvConfig:
    """STATIC env configuration (hashable; changing it triggers recompilation).

    Dynamic per-call values (team sizes, policy params, ratios) live in EnvParams pytrees
    (hhmarl_jax/types.py).
    """
    kind: Literal["lowlevel", "hier"] = "lowlevel"
    agent_mode: Literal["fight", "escape"] = "fight"   # lowlevel only
    level: int = 1                                     # lowlevel curriculum level 1..5
    n_agents_max: int = 2       # array size of the agent team (slots 0..n_agents_max-1)
    n_opps_max: int = 2         # array size of the opponent team
    map_size: float = 0.3       # degrees (0.3 lowlevel, 0.5 hier)
    horizon: Optional[int] = None  # None -> horizons_ll[level-1] (lowlevel) / horizon_hl (hier)
    horizons_ll: Tuple[int, ...] = (150, 200, 300, 350, 400)   # paper: (200,250,300,350,400)
    horizon_hl: int = 500
    ac1: AircraftConfig = AC1_CODE
    ac2: AircraftConfig = AC2_CODE

    # ammunition overrides (spec §4.3); (cannon_s, ac1_missiles)
    ll_opp_ammo_l1_4: Tuple[float, int] = (400.0, 8)   # opponents only, level <= 4
    ll_all_ammo_l5: Tuple[float, int] = (300.0, 6)     # ALL units at level 5
    hl_all_ammo: Tuple[float, int] = (300.0, 8)
    missile_wait_ll: Tuple[int, int] = (7, 17)         # randint inclusive
    missile_wait_hl: Tuple[int, int] = (8, 12)

    # opponents
    l5_opponents: Literal["code", "paper"] = "code"    # code: uniform {L3 fight, L4 fight, L3 escape}
    friend_rule_ll: Literal["pair", "nearest"] = "pair"  # pair = 1<->2, 3<->4 (original, 2v2)

    # rewards (spec §8, §7.2)
    rew_scale: float = 1.0
    glob_frac: float = 0.0
    friendly_kill: bool = True
    friendly_punish: bool = False
    kill_reward: Literal["code", "paper"] = "code"
    killed_penalty: float = -2.0          # paper: 0.0
    friendly_kill_penalty: float = -2.0
    oob_penalty_ll: float = -5.0
    oob_penalty_hl: float = -2.0          # paper: -5.0
    escape_step_reward: Literal["none", "code", "paper"] = "none"  # code flag esc_dist_rew=False
    escape_fire_penalty: float = -0.1

    # commander env (spec §7)
    n_opp_hl: int = 2                     # paper: 3 (code block layout: obs 34->44, actions 3->4)
    hl_situation_reward: Literal["code", "paper", "none"] = "code"
    hl_substep_rule: Literal["code", "paper"] = "code"   # code: <=16 ticks, kill exit, situation after s>10
    hl_max_substeps: int = 16
    hl_situation_after: int = 10          # situation event checked when s > 10

    quirks: EnvQuirks = EnvQuirks()
    record_ticks: bool = False            # hier: return per-sub-tick TickRecord in info (viz)

    # ------------------------------------------------------------------ helpers
    @property
    def n_slots(self) -> int:
        """Total aircraft slots A = n_agents_max + n_opps_max."""
        return self.n_agents_max + self.n_opps_max

    @property
    def episode_horizon(self) -> int:
        if self.horizon is not None:
            return self.horizon
        return self.horizons_ll[self.level - 1] if self.kind == "lowlevel" else self.horizon_hl

    @staticmethod
    def lowlevel(level: int, agent_mode: str = "fight", **kw) -> "EnvConfig":
        return EnvConfig(kind="lowlevel", level=level, agent_mode=agent_mode, map_size=0.3, **kw)

    @staticmethod
    def hier(n_agents: int = 3, n_opps: int = 3, **kw) -> "EnvConfig":
        return EnvConfig(kind="hier", n_agents_max=n_agents, n_opps_max=n_opps, map_size=0.5, **kw)

    @staticmethod
    def paper(base: "EnvConfig") -> "EnvConfig":
        """Apply every paper-vs-code switch (docs/contract.md §3.3). May hit NotImplemented paths."""
        return dataclasses.replace(
            base, horizons_ll=(200, 250, 300, 350, 400), ac1=AC1_PAPER, ac2=AC2_PAPER,
            kill_reward="paper", killed_penalty=0.0, oob_penalty_hl=-5.0,
            escape_step_reward="paper", n_opp_hl=3, hl_situation_reward="paper",
            hl_substep_rule="paper", l5_opponents="paper")


# ======================================================================== networks


@dataclass(frozen=True)
class NetConfig:
    """Network architecture options (spec §10.5)."""
    ll_arch: Literal["sa", "fc"] = "sa"            # fight actor/critic; escape is always "esc"
    cmd_arch: Literal["gru", "sa", "fc"] = "gru"
    # rllib: non-causal MHA over the <=max_seq_len training chunk, T=1 at rollout (faithful).
    # single_step: T=1 at both rollout and training (consistent clean variant).
    sa_mode: Literal["rllib", "single_step"] = "rllib"
    residual_norm: Literal["l2", "layernorm"] = "l2"   # code: F.normalize; paper: Add&Norm
    critic_sizes: Tuple[int, int, int] = (175, 175, 150)  # fight critic v1, v2, v3; paper (300,100,100)
    extra_fc500: bool = False        # paper Fig.5 has FC-tanh 500 after the shared layer
    fc_hidden: int = 500             # FC baseline: D->500 tanh -> SHARED 500 -> out
    shared_layer: bool = True        # one 500x500 tanh layer shared by ac1/ac2 and actor/critic
    mha_heads: int = 2
    max_seq_len: int = 20


# ======================================================================== PPO / RLlib quirks


@dataclass(frozen=True)
class RLlibQuirks:
    """Training-side RLlib behaviour.  All True == rllib_faithful."""
    sa_noncausal_padded_chunks: bool = True     # MHA over whole zero-padded chunk incl. future steps
    loss_mean_over_padding: bool = True         # non-RNN loss branch: plain mean incl. padded rows
    critic_zero_actions_at_gae: bool = True     # V for GAE computed with zero action slots
    shared_layer_sequential_adam: bool = True   # ac1 then ac2 SGD, separate Adam states on shared layer
    no_bootstrap_at_horizon: bool = True        # truncation treated as terminal
    dead_agent_rows_in_loss: bool = True        # dead agents' zero-obs transitions are trained on
    commander_lambda_one: bool = True           # commander GAE lambda = 1.0 (RLlib default)
    adaptive_kl: bool = True                    # kl_coeff *1.5 / *0.5 using last-epoch mean KL
    standardize_advantages: bool = True         # per policy over the whole train batch

    @staticmethod
    def clean() -> "RLlibQuirks":
        return RLlibQuirks(sa_noncausal_padded_chunks=False, loss_mean_over_padding=False,
                           critic_zero_actions_at_gae=False, shared_layer_sequential_adam=False,
                           no_bootstrap_at_horizon=False, dead_agent_rows_in_loss=False,
                           commander_lambda_one=False)


@dataclass(frozen=True)
class PPOConfig:
    """Defaults = train_hetero.py + RLlib 2.4 PPO defaults (spec §10.2)."""
    lr: float = 1e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    gamma: float = 0.99              # paper: 0.95
    gae_lambda: float = 0.95         # commander uses 1.0 when RLlibQuirks.commander_lambda_one
    clip_eps: float = 0.25           # paper: 0.2
    kl_coeff: float = 0.2
    kl_target: float = 0.025         # commander: 0.05
    vf_coeff: float = 1.0
    vf_clip: float = 10.0            # clips the SQUARED value error
    entropy_coeff: float = 0.0
    num_epochs: int = 30
    minibatch_size: int = 256        # rows (agent-steps) per policy
    train_batch_size: int = 2000     # env steps per iteration (commander: 1000)
    grad_clip: Optional[float] = None

    @staticmethod
    def commander() -> "PPOConfig":
        return PPOConfig(gae_lambda=1.0, kl_target=0.05, train_batch_size=1000)


# ======================================================================== training


@dataclass(frozen=True)
class CurriculumConfig:
    """Low-level curriculum + FSP (spec §10.1, §5.2, README procedure)."""
    phases: Tuple[str, ...] = ("L1_fight", "L2_fight", "L3_fight", "L4_fight",
                               "L3_escape", "L5_fight", "commander")
    iters_per_phase: Tuple[Tuple[str, int], ...] = ()   # () -> TrainConfig.iterations for all
    warm_start: bool = True          # fight L(n) restores full train state of L(n-1)
    export_every: int = 50           # FSP snapshot export period (iterations), level >= 3
    fsp_pool_size: int = 1           # 1 = latest snapshot only (faithful); >1 = pool extension
    escape_l5: bool = False          # optional "L5_escape" phase vs L5 fight


@dataclass(frozen=True)
class TrainConfig:
    exp_name: str = "default"
    runs_root: str = "runs"          # run dir = {runs_root}/{exp_name}/{phase}
    seed: int = 0
    num_envs: int = 100              # parallel envs (vmap)
    rollout_len: int = 20            # env steps per env per iteration; num_envs*rollout_len = batch
    iterations: int = 10000          # per phase (code: epochs=10000)
    rllib_faithful: bool = True
    quirks: Optional[RLlibQuirks] = None   # None -> RLlibQuirks() if rllib_faithful else clean()
    ckpt_every: int = 50
    log_every: int = 1
    reward_window: int = 100         # RLlib metrics_num_episodes_for_smoothing
    env: EnvConfig = EnvConfig()
    net: NetConfig = NetConfig()
    ppo: PPOConfig = PPOConfig()
    curriculum: CurriculumConfig = CurriculumConfig()

    def resolved_quirks(self) -> RLlibQuirks:
        if self.quirks is not None:
            return self.quirks
        return RLlibQuirks() if self.rllib_faithful else RLlibQuirks.clean()


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation (spec §7.3).  Default env = HighLevel env (50 km, horizon 500) as evaluation.py."""
    kind: Literal["commander", "fight", "escape"] = "commander"
    env_kind: Literal["hier", "lowlevel"] = "hier"
    n_agents: int = 2
    n_opps: int = 2
    n_episodes: int = 1000
    num_envs: int = 1000             # parallel envs for evaluation
    agent_level: int = 5
    opp_level: int = 4               # commander eval forces 5 (code: eval_hl => both L5)
    opp_fight_ratio: float = 0.75    # hier_opp_fight_ratio/100; set 1.0 for pure-fight eval
    commander_deterministic: bool = True       # explore=False -> argmax
    gru_reset_each_step: bool = True           # evaluation.py quirk: GRU state zeroed each step,
                                               # chained across agents within a step
    seed: int = 0
