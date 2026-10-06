"""Shared constants for hhmarl_jax.  CONTRACT-FROZEN (owner: orchestrator).

Every value mirrors the ORIGINAL code (see docs/spec_original_env.md, section refs in comments).
Changing anything here requires an orchestrator-approved edit of docs/contract.md.
"""

# ---------------------------------------------------------------- map / units (spec §1)
MAP_ORIGIN_LAT = 5.0          # bottom_lat of MapLimits
MAP_ORIGIN_LON = 7.0          # left_lon of MapLimits
KNOTS_TO_MS = 0.514444        # cmano_simulator.py:21
TICK_SECS = 1                 # one sim tick = 1 s
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563

# ---------------------------------------------------------------- aircraft types (spec §2.1)
AC1 = 1                       # Rafale: missiles, agile
AC2 = 2                       # RafaleLong: cannon only, longer range
N_AC_TYPES = 2

# ---------------------------------------------------------------- rocket (spec §2.4)
ROCKET_LIFETIME_S = 10        # removed at life > 10 (before its 12th move)
ROCKET_FUSE_KM = 1.0          # strict "<"
ROCKET_MAX_DEG_SEC = 10.0
# speed (knots) at integer life t = 0..10, exact values of the quadratic interp1d
ROCKET_SPEED_TABLE = (
    500.0, 758.75, 993.3333333333334, 1203.75, 1390.0, 1552.0833333333333,
    1690.0, 1803.75, 1893.3333333333333, 1958.75, 2000.0,
)
ROCKET_SPEED_AFTER = 600.0    # fill_value for t > 30 (never reached)
# [BUG] friendly proximity fuse hard-coded to aircraft id 2 == slot index 1 (agent slot 1)
ROCKET_FUSE_SLOT = 1

# ---------------------------------------------------------------- actions (spec §3)
ACTION_NVEC = (13, 9, 2, 2)   # heading, speed, cannon, rocket ; AC2 uses only the first 3
ACTION_DIM = 4                # actions are always int32[..., 4]; AC2 rows must have a[3] == 0
ACTION_DIM_AC = {AC1: 4, AC2: 3}
LOGITS_DIM = 26               # 13 + 9 + 2 + 2 ; AC2 logits are zero-padded from 24 to 26
LOGITS_DIM_AC = {AC1: 26, AC2: 24}
HEADING_STEP_DEG = 15.0       # new heading = heading + (a0 - 6) * 15
SPEED_MIN_CMD = 100.0         # set_speed(100 + (max_speed - 100) / 8 * a1)

# ---------------------------------------------------------------- observations (spec §6)
OBS_FIGHT = {AC1: 26, AC2: 24}
OBS_ESCAPE = {AC1: 30, AC2: 29}
OBS_FIGHT_MAX = 26            # low-level obs arrays are padded to this (fight)
OBS_ESCAPE_MAX = 30           # ... and to this (escape)
# actor input slices (models/ac_models_hetero.py); (start, stop) on the UNPADDED obs
FIGHT_OWN_SLICE = {AC1: (0, 12), AC2: (0, 10)}            # rest = obs[own_stop:]
ESC_SLICES = {                                              # own, 2 opps, friend
    AC1: ((0, 7), (7, 25), (25, 30)),
    AC2: ((0, 6), (6, 24), (24, 29)),
}

# centralized critic input, flattened in RLlib Dict order [act_own | act_other | obs_own | obs_other]
CRITIC_FIGHT_DIM = 57         # 4 + 3 + 26 + 24 (both types)
CRITIC_ESCAPE_DIM = 66        # 4 + 3 + 30 + 29 (both types)

# ---------------------------------------------------------------- commander (spec §6.3, §7)
N_OPP_HL = 2                  # code HEAD; paper uses 3 (see config.HierConfig.n_opp_hl)
N_FRI_HL = 2
CMD_HIDDEN = 200              # GRU hidden size (actor and critic each)


def cmd_obs_dim(n_opp_hl: int = N_OPP_HL) -> int:
    """4 own + 10 per sensed opponent + 5 per sensed friend (2 friends)."""
    return 4 + 10 * n_opp_hl + 5 * N_FRI_HL


def cmd_num_actions(n_opp_hl: int = N_OPP_HL) -> int:
    """0 = escape, k = attack k-th nearest opponent."""
    return 1 + n_opp_hl


def cmd_critic_dim(n_opp_hl: int = N_OPP_HL, n_team: int = 3) -> int:
    """[act_own | act_o1 | act_o2 | obs_own | obs_o1 | obs_o2] (actions scaled by 1/(A-1))."""
    return n_team * (1 + cmd_obs_dim(n_opp_hl))


CMD_OBS_DIM = cmd_obs_dim()          # 34
CMD_NUM_ACTIONS = cmd_num_actions()  # 3
CMD_CRITIC_DIM = cmd_critic_dim()    # 105

# ---------------------------------------------------------------- misc
LEVELS = (1, 2, 3, 4, 5)
MODE_FIGHT = 0
MODE_ESCAPE = 1
