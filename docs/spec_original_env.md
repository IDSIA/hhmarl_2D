# Specification of the ORIGINAL HHMARL-2D code (IDSIA/hhmarl_2D)

Purpose: an implementation-ready description of the reference code for a JAX port aiming at numerical and behavioural parity. File:line citations refer to this repo (`envs/`, `warsim/`, `models/`, `config.py`, `train_*.py`, `evaluation.py`). RLlib internals are cited from the **ray 2.4.0** wheel (`ray/rllib/...`), the version pinned in `README.md:17`.

Notation: `N = num_agents`, `M = num_opps`, `T = N + M` (total aircraft). Aircraft ids are `1..N` (agents, "blue") then `N+1..T` (opponents, "red"). Rocket ids start at `T+1` and are never reused (`warsim/simulator/cmano_simulator.py:104-108`).

Flags: **[BUG]** behaviour that looks unintended but has to be reproduced for parity. **[AMBIG]** unclear or version-dependent behaviour. **[NONDET]** a source of non-reproducibility. **[PORT]** a recommendation for the JAX port.

---

## 0. Quick reference

| Item | Low-level (`LowLevelEnv`) | High-level (`HighLevelEnv`) |
|---|---|---|
| Config mode | `Config(0)` | `Config(1)` (train) / `Config(2)` (eval) |
| Default N vs M | 2 vs 2 (`config.py:20-22`) | 3 vs 3 |
| Map size (deg) | 0.3 (about 33 km) (`config.py:38`) | 0.5 (about 55 km) |
| Horizon | L1 150, L2 200, L3 300, L4 350, L5 400 sim ticks (`config.py:94-96`) | 500 sim ticks (`config.py:98`) |
| Tick | 1 s (`cmano_simulator.py:80`) | 1 s; one commander step = 1..16 ticks |
| Obs AC1 / AC2 (fight) | 26 / 24 | – |
| Obs AC1 / AC2 (escape) | 30 / 29 | – |
| Commander obs | – | 34 = 14 + 10·N_OPP_HL, with N_OPP_HL = 2 (`env_hier.py:20-25`) |
| Actions AC1 | MultiDiscrete [13, 9, 2, 2] | – |
| Actions AC2 | MultiDiscrete [13, 9, 2] | – |
| Commander action | – | Discrete(3): 0 = escape, k = attack the k-th nearest opp |
| PPO batch | 2000 env steps, minibatch 256 | 1000 env steps, minibatch 256 |

Aircraft types: **AC1 = `Rafale`** (has missiles), **AC2 = `RafaleLong`** (cannon only, longer range).

---

## 1. Geometry, units, coordinate conventions

* Positions are geodetic `(lat, lon)` in degrees on WGS84. Altitude is fixed at 10 000 m and never used (`env_base.py:562-564`).
* Map rectangle `MapLimits(left_lon=7.0, bottom_lat=5.0, right_lon=7.0+map_size, top_lat=5.0+map_size)` (`env_base.py:43`). Real extent: 0.3° gives 33.27 km (lon) × 33.17 km (lat); 0.5° gives 55.4 km × 55.3 km.
* `relative_position(lat, lon)` returns **`(lat_rel, lon_rel)`, each clipped to [0, 1]** (`warsim/utils/map_limits.py:37-40`). In every observation the code writes `x, y = relative_position(...)`, so the first feature "x" is the **normalised latitude (north)** and "y" is the **normalised longitude (east)**. `_sample_state` uses the opposite convention: `x` = lon and `y` = lat, fed to `Position(y, x, ...)` (`env_base.py:562`).
* `in_boundary`: `left_lon <= lon <= right_lon and bottom_lat <= lat <= top_lat` (inclusive) (`map_limits.py:47-48`).
* Heading: compass bearing in degrees, 0 = north, clockwise, range [0, 360).
* Speed in knots; `knots_to_ms = 0.514444` (`cmano_simulator.py:21`).
* Kinematics and weapons use **geodesic** distance and initial azimuth: `Geodesic.WGS84.Inverse` and `Direct` (`warsim/utils/geodesics.py`).
* Observation and reward geometry uses **planar degree space** (lon as x, lat as y), with no cos(lat) correction:
  * `_distance(a, b)` = `hypot(Δlon, Δlat)` in degrees. Normalised form = `d / sqrt(2·map_size²)` (`env_base.py:434-439`). The normalised value is not clipped and can slightly exceed 1 only theoretically.
  * `_focus_angle(a, b)` (ATA, antenna train angle) = angle between a's heading unit vector `(cos((90−h)%360·π/180), sin(...)) = (sin h, cos h)` in (lon, lat) and the vector a→b `(Δlon, Δlat)`: `acos(clip(dot/(|u|·|v| + 1e-10), −1, 1))` in degrees in [0, 180]. Normalised = `/180`, clipped to [0, 1] (`env_base.py:424-432`). Coincident positions give acos(0) = 90°.
  * `_aspect_angle(a, b, norm=True)` = `clip((180 − focus(a, b))/180, 0, 1)`, which is just `1 − focus_norm(a, b)` (`env_base.py:441-446`).
  * `_heading_diff(a, b)` = angle between the two heading vectors / 180 (`env_base.py:448-456`). Symmetric.
* `signed_heading_diff(actual, desired)` = `desired − actual` wrapped into [−180, 180] (`warsim/utils/angles.py:22-29`). Note that ±180 can both occur. `sum_angles` normalises to [0, 360) (`angles.py:10-19`).

**[PORT] Geodesic approximation.** I checked this numerically with geographiclib in the region lat 5–5.5, lon 7–7.5. A local ellipsoidal flat-earth step with meridional radius `M(lat)` and normal radius `N(lat)` evaluated at the mid-latitude (`dlat = d·cos h / M`, `dlon = d·sin h / (N·cos lat_mid)`; WGS84 a = 6378137, f = 1/298.257223563) has these errors:
* max position error ≤ 9 mm per step for d ≤ 1.1 km;
* distance error ≤ 0.11 m over the whole map;
* initial-bearing error ≤ 0.023° (meridian convergence; it can be corrected by `−Δlon·sin(lat_m)/2`).

These errors are negligible against the thresholds (cannon cones of ±5° and ±3.5°, 1 km rocket fuse). float32 has a resolution of about 5 cm at lon 7.5°. Positions can therefore be stored in float32, but it is preferable to store offsets from (5°, 7°), or to use float64 on CPU parity tests.

---

## 2. Aircraft and weapon models

### 2.1 Constants

| | `Rafale` (AC1) `warsim/simulator/ac1.py:24-36` | `RafaleLong` (AC2) `ac2.py:23-32` |
|---|---|---|
| max turn rate | 5 °/s | 3.5 °/s |
| speed range | [0, 900] kn (`max_speed` = 900) | [0, 600] kn (`max_speed` = 600) |
| max accel | 35 kn/s | 28 kn/s |
| cannon range | 2.0 km (strict `<`) | 4.5 km |
| cannon cone width | 10° (±5°, `<=`) | 7° (±3.5°) |
| cannon ammo (class default) | 200 s | 200 s |
| burst length | 5 s | 3 s |
| hit prob per burst | 0.75 → **0.15 per tick** | 0.9 → **0.30 per tick** |
| missiles (class default) | 5 | 0 (`rocket_max` = 0) |
| missile range | 111 km (never binding) | – |
| missile "radar" width | 120° | – |

Per-scenario overrides of ammunition: see §4.3. They overwrite both `cannon_max` and `rocket_max`, and those are the normalisers used in the observations.

### 2.2 Commands (set before the tick)
* `set_heading(h)` requires `0 <= h < 360` or raises; it only stores `new_heading` (`ac1.py:58-61`).
* `set_speed(s)` requires `min <= s <= max` or raises; it stores `new_speed` (`ac1.py:63-67`).
* `fire_cannon()`: `cannon_current_burst_secs = min(cannon_remain_secs, burst_time)` (`ac1.py:69-70`). Calling it again during a burst **resets** the burst to its full length. Calling it with no ammo sets the burst to 0, which does nothing.
* `fire_missile(ag_unit, opp_unit, sim)` (AC1 only, `ac1.py:72-79`): launches only if `actual_missile is None and missile_remain > 0 and dist_km <= 111 and _angle_in_radar_range(bearing(self→opp))`. On launch: `Rocket(position=copy(self.pos), heading=self.heading, firing_time=sim.utc_time, target=opp, source=ag_unit)` is added to the sim, `actual_missile` is set, and `missile_remain -= 1`.
* **[BUG] Radar cone is offset.** `_angle_in_radar_range(b)`: `delta = |signed_heading_diff(heading + 60, b)|; return int(delta) <= 60` (`ac1.py:144-146`). The valid bearings are therefore **[heading, heading + 120]**, i.e. from dead-ahead to 120° on the right side, and not ±60°. `int()` truncation extends this by just under 1° on each side: delta < 61.

### 2.3 Aircraft update inside `do_tick` (`ac1.py:81-133`, `ac2.py:68-107`), in this exact order
1. **Heading.** If `heading != new_heading`: `δ = signed_heading_diff(heading, new_heading)`. If `|δ| <= max_deg`, snap to the target; otherwise `heading += ±max_deg` (sign of δ, with δ = 0 counted as +), then `heading %= 360`.
2. **Speed.** Same rule with `max_knots_sec`.
3. **Cannon.** If `burst > 0`:
   * `burst = max(burst − 1, 0)` and `cannon_remain = max(cannon_remain − 1, 0)`;
   * then, for every unit `u` in the *current* `sim.active_units` (a snapshot taken now) with `u.id != self.id` and `u.id <= N + M` (aircraft only), subject to the friendly filter:
     * filter = `friendly_check or (group == "agent" and u.id >= N+1) or (group == "opp" and u.id <= N)`. With `friendly_kill=True` (the default, `config.py:45`) **teammates can be hit too**;
     * if `dist_km(self, u) < range` and `|signed_heading_diff(self.heading, bearing(self→u))| <= width/2`, then with probability `p_hit / burst_time` (`sim.rnd_gen.random() < p`): **remove u immediately** and emit `UnitDestroyedEvent(origin=self, killer=self, destroyed=u)`.
   * Because the decrement happens before the check, a full burst gives exactly `burst_time` checking ticks. The kill probability over a whole burst on a target that stays in the cone is 1 − 0.85⁵ = 0.556 (AC1) and 1 − 0.7³ = 0.657 (AC2).
   * The cone test uses the shooter's heading *after* step 1 and the shooter's position *before* its own move. Units with a lower id have already moved this tick; units with a higher id have not.
4. **Missile steering (AC1 only).** If `actual_missile`:
   * if the rocket no longer exists in the sim, set `actual_missile = None`;
   * else `rocket.new_heading = clip(rocket.heading · U(0.95, 1.05), 0, 359)`, using the **global Python `random`**.
   * **[BUG]** The missile is **unguided**: the target-bearing guidance is commented out (`ac1.py:119-125`). The noise is multiplicative, so it is about ±0.5° near heading 10° and ±17.5° near heading 350°.
5. **Move.** If `speed > 0`: geodesic direct along `heading` for `speed · 0.514444 · 1 s` metres (`cmano_simulator.py:65-72`). This is semi-implicit: the new heading and speed are used.

### 2.4 Rocket (`warsim/simulator/rocket_unit.py:14-73`)
* `speed_profile = interp1d([0, 10, 20, 30], [500, 2000, 1400, 600], kind='quadratic', fill_value=(500, 600))`. Only integer life times 0..10 are ever used. Exact values (computed with scipy):

| life t (s) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speed (kn) | 500 | 758.75 | 993.3333 | 1203.75 | 1390 | 1552.0833 | 1690 | 1803.75 | 1893.3333 | 1958.75 | 2000 |

  This is a single quadratic B-spline with knots [0, 0, 0, 15, 30, 30, 30] and coefficients [500, 2531.25, 1156.25, 600]. On [0, 15] it equals `500 + 270.4167·t − 7.0417·t²`. Total straight-line reach is about 8.10 km over 11 moves.
* `max_deg_sec = 10`. The heading update has **no `% 360`** (`rocket_unit.py:61-67`), so the heading can temporarily be < 0 or ≥ 360. The next steering `clip(...)` brings it back. **[BUG]** minor.
* Update order per tick:
  1. **Target hit.** If `dist_km(rocket, target) < 1 and target exists`: remove the rocket and the target, emit `UnitDestroyedEvent(origin=rocket, killer=source, destroyed=target)`, return. The distance uses the rocket's position **before** its own move this tick and the target's position **after** its move (aircraft always update before rockets).
  2. **[BUG] Friendly proximity fuse hard-coded to id 2.** If `friendly_check`: `fid = 1 if source.id == 2 else 2`. Since aircraft 2 is always a RafaleLong (it never fires), **every rocket, from agent 1 or from any opponent, detonates on aircraft id 2 (agent 2) if it is within 1 km**. It removes both and emits `UnitDestroyedEvent(rocket, source, unit2)`. Consequences:
     * opponent rockets get a second target, agent 2;
     * agent 1's rocket can kill agent 2 at launch if agent 2 is within 1 km;
     * opponents' own teammate (id 4) is never protected or endangered.
  3. **End of life.** `life = (sim.utc_time − firing_time).seconds`. If `life > 10`: remove the rocket (no event), return.
  4. Heading turns toward `new_heading` by at most 10°.
  5. `speed = profile(life)`.
  6. Geodesic move.

  A rocket fired during the action phase gets its first update in the same tick with life = 0. It moves at life = 0..10 (11 moves) and is removed at its life = 11 update, before any move. The shooter's `actual_missile` is cleared one tick later, at the shooter's next update (§2.3.4). The "shot" observation flag therefore stays 1 for one extra step.
* The rocket's `source` keeps a reference to the shooter object, even after the shooter is dead. A dead shooter is no longer updated, so it stops steering: the rocket keeps its last `new_heading` and flies on. Kills by such a rocket still emit events with `killer = dead shooter`.

### 2.5 Simulator tick (`cmano_simulator.py:138-157`)
* `do_tick` iterates over `list(active_units.values())`, which is a **snapshot in insertion (= id) order**: aircraft 1..T, then rockets in launch order.
* **[BUG] Dead units still act within a tick.** A unit removed earlier in the same tick (for example by the cannon of a lower id) is **still updated**. It can still fire its cannon burst and kill, steer its rocket and move. Its rocket's own checks are skipped only if the rocket itself was removed. Mutual kills within a tick are therefore possible, and ordering matters: a lower id shoots first.
* `utc_time += 1 s` after all updates.
* `CmanoSimulator(...)` is re-created on every `reset` (`env_base.py:75`). `rnd_gen = random.Random(None)` is seeded from OS entropy on every reset. **[NONDET]** Cannon hits cannot be reproduced even if the global `random` is seeded.

**[PORT]** Implement the tick as a sequential `lax.scan` / unrolled loop over the fixed aircraft slots, followed by a scan over rocket slots in launch order. There are at most one live rocket per AC1 at a time, so `N_AC1 + M_AC1` rocket slots suffice, and they must be ordered by launch id. Keep a per-slot "alive at tick start" mask that controls who is updated, separate from the "alive now" mask that controls who can be hit.

---

## 3. Action semantics (`env_base.py:214-238`, `_take_base_action`)

For a learned or FSP-controlled aircraft with action vector `a`:
* `a[0] ∈ 0..12`: `set_heading((heading + (a[0] − 6)·15) % 360)`. This is relative to the *current* heading, from −90° to +90° in 15° steps. Because the turn rate is capped at 5 or 3.5 °/s and the command is re-issued every tick, **any a[0] ≠ 6 is effectively "turn at max rate" toward that side**. Only the sign matters, apart from snapping when |Δ| ≤ max_deg, which never happens for a nonzero 15° multiple.
* `a[1] ∈ 0..8`: `set_speed(100 + ((max_speed − 100)/8)·a[1])`. AC1 gives {100, 200, …, 900}; AC2 gives {100, 162.5, …, 600}.
* `a[2] ∈ {0, 1}`: if 1 and `cannon_remain_secs > 0`, call `fire_cannon()`.
  * LowLevel escape mode only, agents only: if additionally `cannon_remain_secs < 90`, the reward gets −0.1. This is tested before the tick decrement.
* `a[3] ∈ {0, 1}` (AC1 only): if 1 and `opp_id` (the current target) is truthy and `missile_remain > 0 and not actual_missile and missile_wait == 0`:
  * call `fire_missile(...)`. This attempt can fail silently because of the radar cone;
  * **[BUG] The cooldown is set even if the launch failed**: `missile_wait = randint(7, 17)` in LowLevel and `randint(8, 12)` in HighLevel (inclusive bounds; global `random`);
  * LowLevel escape mode only: if `missile_remain < 3` (after the possible decrement), the reward gets −0.1.
* Cooldown tick: `if missile_wait > 0 and not actual_missile: missile_wait −= 1`. This runs in the same call, so a failed launch immediately counts down by 1. While a missile is in the air, the cooldown is frozen.
* The `missile_wait` cooldown is **only decremented inside `_take_base_action`**. Scripted opponents (L1–L3) never call it, see §5.1.

Target `opp_id`:
* LowLevel: `opp_to_attack[i]` = the nearest opponent, computed in the last `state()` call (agents) or in the `lowlevel_state` call just before acting (FSP opponents).
* HighLevel: see §7.

---

## 4. Episode reset and initial-state sampling

### 4.1 Reset flow (`env_base.py:62-77`, `env_hetero.py:53-60`, `env_hier.py:44-47`)
* `steps = 0`, `alive_agents = alive_opps = 0`, `missile_wait[i] = 0` and `opp_to_attack[i] = None` for i in 1..T. The L3 scripted escape flag and timer are reset.
* A new simulator is created, then `_reset_scenario(mode)`.
* LowLevel L5 fight only: `k = randint(3, 5)` (uniform over {3, 4, 5}). The opponent policy set is `policies[k]`, and `opp_mode = "escape"` if k == 5, else "fight" (`env_hetero.py:55-59`).
* `reset` returns `state(), {}`. The `seed` argument is ignored. **[NONDET]** The global Python `random` is never seeded.

### 4.2 `_reset_scenario` (`env_base.py:551-585`)
* `r = randint(1, 2)` chooses the sides (one draw per episode).
* Loop: group "agent" with i = 0..N−1, then "opp" with i = 0..M−1:
  * `(x = lon, y = lat, a = heading) = _sample_state(group, i, r)`;
  * aircraft type: `ac = i + 1 if i <= 1 else randint(1, 2)`. Slot 0 is AC1 and slot 1 is AC2 in both teams; slots ≥ 2 are uniformly random. 2v2 is therefore deterministic: ids 1 and 3 are AC1, ids 2 and 4 are AC2;
  * initial speed = **0 if `args.level <= 2 and group == "opp"`**, else 100 kn. `new_speed` = speed and `new_heading` = heading;
  * **[BUG]** In HighLevel mode `args.level` keeps its argparse default of 1 (`config.py:17`; nothing overrides it for modes 1 and 2), so **commander-level opponents also start at speed 0**. The low-level policies speed them up at most 35 or 28 kn/s.

Low-level `_sample_state(group, i, r)` (`env_base.py:489-549`). `uniform` is `random.uniform` and `randint` is inclusive.

| level | agent, r = 1 | agent, r = 2 | opp, r = 1 | opp, r = 2 |
|---|---|---|---|---|
| 1 | lon U(7.12, 7.14), lat U(5.10+0.1i, 5.11+0.1i), hdg randint(30, 150) | lon U(7.16, 7.17), lat same, hdg randint(200, 330) | lon U(7.16, 7.17), lat U(5.10+0.1i, 5.11+0.1i), **hdg 0** | lon U(7.12, 7.14), lat same, **hdg 0** |
| 2 | lon U(7.08, 7.13), lat U(5.08+0.1i, 5.13+0.1i), hdg randint(0, 180) | lon U(7.18, 7.23), lat same, hdg randint(180, 359) | lon U(7.18, 7.23), lat U(5.08+0.1i, 5.13+0.1i), hdg randint(0, 359) | lon U(7.08, 7.13), lat same, hdg randint(0, 359) |
| ≥3 | lon U(7.07, 7.12), lat U(5.09+0.1i, 5.12+0.1i), hdg randint(0, 270) | lon U(7.18, 7.23), lat same, hdg randint(90, 359) | lon U(7.18, 7.23), lat U(5.09+0.1i, 5.12+0.1i), hdg randint(0, 359) | lon U(7.07, 7.12), lat same, hdg randint(0, 359) |

High-level `_sample_state` (`env_hier.py:226-250`). With `K = N` for agents and `K = M` for opponents:
* agent, r = 1: lon U(7.07, 7.22), lat U(5.07 + i·0.4/K, 5.12 + i·0.4/K), hdg randint(0, 359). r = 2 is the same with lon U(7.28, 7.43).
* opp: r = 1 gives lon U(7.28, 7.43); r = 2 gives lon U(7.07, 7.22). Same lat formula, hdg randint(0, 359).

RNG call order per unit: `uniform(x)`, `uniform(y)`, `randint(a)` (if applicable), then `randint(1, 2)` for the type if i ≥ 2.

### 4.3 Ammunition overrides (`env_base.py:566-578`)
* LowLevel, level ≤ 4, **opponents**: cannon 400 s; AC1 missiles 8. Agents keep the class defaults: cannon 200 s and 5 missiles (AC1).
* LowLevel, **level 5: all units (agents and opponents)**: cannon 300 s and AC1 missiles 6. The condition is `elif self.args.level == 5`, with no group test.
* HighLevel: all units have cannon 300 s and AC1 missiles 8.
* `cannon_max` and `rocket_max` are overwritten together, so the observation normalisers change too.

---

## 5. Opponent policies (LowLevel)

Dispatch (`env_hetero.py:160-182`), iterated over i = 1..T in id order for existing units:
* agents (i ≤ N): use the RL action;
* opponents with level ≥ 4: FSP policy `_policy_actions(opp_mode, i, u)` followed by `_take_base_action`;
* opponents with level 1, 2 or 3: scripted.

### 5.1 Scripted opponents
`steps` has already been incremented, so it is 1 at the first step.
* **L1** (`env_hetero.py:118-123`): static (speed 0, heading 0, no `set_*`). If AC1 and `not actual_missile and steps % 40 in {0, 1, 2} and randint(0, 1) and missile_wait == 0`: fire a missile at the nearest agent (attempt) and set `missile_wait = 5`.
  * **[BUG]** `missile_wait` is never decremented for scripted opponents, so there is **at most one launch attempt per opponent per episode**, at steps 1, 2, 40, 41, 42 and so on until the first coin success.
  * With heading 0, the radar cone [0°, 120°] only contains agents to the north-east. In practice launches happen only for r = 2, where the agents are east of the opponents.
  * The function reads `self.missile_wait[i]` through a closure over the loop variable; it is the same as `unit_id`.
* **L2** (`env_hetero.py:125-136`): every step `fire_cannon()`. The 400 s of ammo outlast the 200-step horizon. If `steps <= 5 or steps % randint(35, 45) <= 5`: `set_heading((h ± 90) % 360)` with the sign from `randint(0, 1)`, and `set_speed(100 + randint(0, 4)·75)`. Missile logic is the same as L1, with one attempt per episode.
* **L3** (`env_hetero.py:138-158`, `_hardcoded_opp` at `247-271`, `_escaping_opp` at `227-245`):
  * Shared env-level flag `hardcoded_opps_escaping` and timer `opps_escaping_time`. **[BUG]** Both are evaluated and updated *per opponent*:
    * if `steps % 60 == 0 and not escaping`: `escaping = bool(randint(0, 1))`, and if true `timer = int(U(20, 30))`. Opponent 4 can re-roll if opponent 3's roll failed;
    * while escaping, every alive opponent calls `_escaping_opp` and decrements the shared timer, so the timer runs down about 2× faster with 2 opponents alive. When the timer reaches ≤ 0, the flag is cleared.
  * `_escaping_opp`: `(y = lat_rel, x = lon_rel) = relative_position`. Quadrant gives the heading `int(U(...))`: south-west → [30, 60), south-east → [300, 330), north-west → [120, 150), north-east → [210, 240). The opponent heads back toward the centre. Speed `int(U(300, 600))`, cannon fire = `randint(0, 1)`, no missile.
  * `_hardcoded_opp` (pursuit), with `d` = normalised distance to the nearest agent and `focus` = ATA in **degrees** from the opponent to that agent:
    * `speed = int(U(100, 400))`;
    * `sign = _correct_angle_sign(opp, agent)`: `+1` if the agent is clockwise (right) of the heading line, using `round(sin h, 3)` and `round(cos h, 3)`; else `−1` (`env_base.py:464-487`);
    * `r = U(0.7, 1.3)`;
    * if `d > 0.008 and focus > 4`: `heading = (heading + r·sign·focus) % 360`. This is a float, and the turn is still rate-limited;
    * if `d > 0.05`: `speed = int(U(500, 800))` if focus < 30, else `int(U(100, 500))`;
    * `fire = d < 0.03 and focus < 10`; `fire_missile = d < 0.09 and focus < 5`;
    * AC2: `speed = clip(speed, 0, 600)`. AC1 opponents can fly up to 800.
    * In km, for map 0.3 (normaliser 0.4243°): 0.008 ≈ 0.38 km, 0.03 ≈ 1.41 km, 0.05 ≈ 2.35 km, 0.09 ≈ 4.2 km.
  * Then `set_heading`, `set_speed`, `fire_cannon()` if fire. If `fire_missile and opp and not actual_missile and missile_wait == 0 and AC1`: attempt a launch and set `missile_wait = 10`. **[BUG]** As in L1/L2, the cooldown never decrements, so there is one attempt per episode.
* L3 opponents are also the opponents for **escape L3** training.

### 5.2 Fictitious self-play (L4, L5) (`env_base.py:312-398`, `env_hetero.py:48-51`)
* Policies are frozen PyTorch models exported every 50 training iterations from level ≥ 3 runs into `policies/L{level}_AC{1|2}_{mode}.pt` (`train_hetero.py:98-107`). They are loaded **once per env instance at construction**.
  * L4 fight: opponents = L3 fight.
  * L5 fight: per episode, uniform over {L3 fight, L4 fight, L3 escape}.
  * L5 escape (`config.py:86-90` sets level 5 once `L3_escape_2-vs-2` exists): opponents = L5 fight.
* Inference (`_policy_actions`):
  * build the obs with `lowlevel_state(policy_type, agent_id=i)`. For opponents, the "friend" is `3 ↔ 4` (`env_hetero.py:71-75`). The call also sets `opp_to_attack[i]` = nearest agent;
  * feed `{"obs_1_own": obs[1×D], "obs_2": zeros, "act_1_own": zeros, "act_2": zeros}` with `seq_lens = [1]` (`env_base.py:357-396`);
  * **actions = per-component argmax of the logits (deterministic)**, split [13, 9, 2, 2] or [13, 9, 2] (`env_base.py:373-382`).
* **[BUG] Ordering subtlety.** Opponent observations are computed *inside* the action loop, after agents 1..N have already called `fire_cannon`/`fire_missile` this step. The opponent therefore observes the agents' *current-step* shot flags (`cannon_current_burst_secs > 0`, `actual_missile`) and its own state. Positions are still pre-tick.

---

## 6. Observations

All observations are float32 and lie in [0, 1]. The observation for a dead agent, or for an agent with no living opponent, is all zeros (`env_hetero.py:97-100`; `env_hier.py:88-91`). `heading_norm = clip((heading % 359)/359, 0, 1)`. **[BUG]** It uses 359, not 360, so headings in [359, 360) map near 0. `speed_norm = clip(speed/max_speed, 0, 1)` uses the observed unit's own `max_speed` (900 or 600).

In the low-level env, `opp` = the nearest living opponent by normalised planar distance (stable sort, `env_base.py:400-422`), and `fri` = the fixed partner (`1 ↔ 2`, `3 ↔ 4`, `env_hetero.py:71-75`). `dist` = normalised planar distance.

### 6.1 Fight obs (`fight_state_values`, `env_base.py:111-135`; `opp_ac_values("fight")`, `185-212`; `friendly_ac_values`, `166-183`)

AC1, 26 dims:

| idx | feature |
|---|---|
| 0 | own lat_rel |
| 1 | own lon_rel |
| 2 | own speed_norm |
| 3 | own heading_norm |
| 4 | focus(own→opp)/180 |
| 5 | aspect(opp, own) = 1 − focus(opp→own)/180 |
| 6 | heading_diff(own, opp)/180 |
| 7 | dist(own, opp) |
| 8 | cannon_remain/cannon_max |
| 9 | missile_remain/rocket_max |
| 10 | 1 if missile_wait == 0 |
| 11 | own shot = 1 if (actual_missile is not None) or burst > 0 |
| 12–20 | **opp block**: opp lat_rel, opp lon_rel, opp speed_norm, opp heading_norm, heading_diff(opp, own), focus(opp→own)/180, aspect(own, opp) = 1 − focus(own→opp)/180, dist (same value as idx 7), opp shot flag (burst > 0, or actual_missile for AC1) |
| 21–25 | **friend block**: fri lat_rel, fri lon_rel, focus(own→fri)/180, focus(fri→own)/180, dist(own, fri). All zeros if the friend is dead |

AC2, 24 dims: idx 0–8 as for AC1; 9 = own shot (burst > 0); 10–18 = opp block; 19–23 = friend block.

### 6.2 Escape obs (`esc_state_values`, `env_base.py:137-164`)

AC1, 30 dims: 0 lat, 1 lon, 2 speed, 3 heading, 4 cannon_remain/max, 5 missile_remain/max, 6 own shot. Then **2 opponent blocks, nearest first**, of 9 each at 7–15 and 16–24. Each block is: lat, lon, speed, heading, heading_diff(opp, own), **focus(own→opp)/180, focus(opp→own)/180**, dist, opp shot. Missing opponents are zero-padded. 25–29 = friend block.

AC2, 29 dims: 0–3 as above, 4 cannon, 5 shot, 6–23 opponents, 24–28 friend.

### 6.3 Commander obs (`HighLevelEnv.state`, `env_hier.py:49-98`), 34 dims, for agents 1..N only
* 0–3: own lat, lon, speed_norm, heading_norm.
* 4–23: `N_OPP_HL = 2` opponent blocks of 10, nearest first, zero-padded: lat, lon, speed, heading, heading_diff(opp, own), focus(own→opp)/180, focus(opp→own)/180, aspect(own, opp), aspect(opp, own), dist. There is no shot flag. The aspects are redundant (1 − focus).
* 24–33: 2 friend blocks of 5 (nearest friends first, zero-padded), in the same format as §6.1.
* Side effects of `state()`: `opp_to_attack[agent]` = the **full sorted list** of all living opponents as `[id, dist_norm, dist_deg]`. Only the first 2 appear in the obs, but **all** of them are stored. For each living opponent, `opp_to_attack[opp]` = the full sorted list of living agents.

### 6.4 Low-level obs inside the HighLevel env (`env_hier.py:100-112`)
* Fight: `fight_state_values(i, unit, opp_to_attack[i][commander_action[i] − 1], fri_id)`. Escape: `esc_state_values(i, unit, opp_to_attack[i], fri_id)`. `fri_id` = the **nearest** living teammate, recomputed every sub-step.
* **[BUG] Stale distance.** The `opp` entries come from `opp_to_attack`, which is computed once per commander step. The distance features (fight idx 7 and the dist in the opp block; escape per-opponent dist) are therefore **frozen at the value from the start of the commander step** for up to 16 ticks. All other features are live. In escape mode the *order* of opponents is also frozen.

---

## 7. High-level (commander) environment

* Spaces: obs Box(34) per agent; action Discrete(3) (`env_hier.py:36-37`). Low-level policies are loaded at construction (`_get_policies("HighLevel")`, `env_base.py:332-346`):
  * `fight_{1,2}` = `L{eval_level_ag}` (default **L5**, `config.py:51`);
  * `escape_{1,2}` = L5 escape if present, else L3 escape;
  * if `eval_hl = False` (evaluation only), additionally `fight_{1,2}_opp` = `L{eval_level_opp}`.
  * The **same frozen low-level policies drive both agents and opponents.**

### 7.1 `step(commander_actions)` → `_take_action` (`env_hier.py:114-140`)
1. `_action_assess(rewards = {})` (`env_hier.py:142-190`), for i = 1..T:
   * **Living agent:** `rewards[i] = 0`.
     * If action > 0: `opp_id = opp_to_attack[i][action − 1][0]`. On IndexError (the slot was zero-padded): `opp_id = None`, `commander_actions[i] = 1`, `rewards[i] = −0.1`. Otherwise, if `hier_action_assess` (default True): `rewards[i] = 0.1` if `dist_deg(i, opp) < 0.1 and focus(i→opp) < 15° and focus(opp→i) > 40°`, else 0.
     * If action = 0 (escape) and `hier_action_assess`: with `c` = nearest opponent, `rewards[i] = 0.1` if `dist_deg(c, i) < 0.1 and focus(c→i) < 15° and focus(i→c) > 40°`.
   * **Living opponent:**
     * `(p, q) = Fraction(hier_opp_fight_ratio, 100).limit_denominator().as_integer_ratio()`; the default 75 gives (3, 4);
     * `fight = random.choices([0, 1], weights = [q − p, p])`, so P(fight) = 0.75;
     * if fight: with `n` = number of living agents, if `n > 1` and `random.choices([0, 1], weights = [1, 3])` → 1 (**prob 0.75**), then `ag = randint(2, n)` (a *non-nearest* agent); else `ag = 1` (the nearest);
     * else `ag = 0` (escape);
     * `commander_actions[i] = ag`. This mutates the dict passed in by RLlib.
   * **Dead unit:** agents get `rewards[i] = 0`; `commander_actions[i] = None` for all dead units.
2. Sub-step loop: `s = 0`; `while s <= 15 and not kill_event and not situation_event`:
   * for i = 1..T existing, in id order: `policy = "escape" if ca[i] == 0 else "fight"`; actions = FSP argmax (§5.2) on the obs of §6.4; then `_take_base_action("HighLevel", u, i, opp_to_attack[i][ca[i] − 1][0], actions)`.
     * **[BUG]** For escape, `ca[i] − 1 = −1`, so the missile target is the **farthest** opponent in the frozen list.
   * `rewards, kill_event = _get_rewards(rewards, do_tick())`. `kill_event` is True if any OOB removal or *any* destroy event happened (including friendly kills).
   * If `s > 10`: `situation_event = _surrounding_event()` (`env_hier.py:192-208`). It is True if any living (agent, opponent) pair has `dist_deg < 0.1` and (`focus(ag→opp) < 15°` or `focus(opp→ag) < 15°`).
   * `s += 1; steps += 1`.
   * One commander step is therefore 1..16 ticks. The situation check starts after the 12th tick (s = 11). `min_sub_steps` is fixed at 10 (the random version is commented out, `env_hier.py:120`).
3. Horizon 500 is checked only between commander steps, so an episode can run to `steps = 515`.

### 7.2 HighLevel `_get_rewards` (`env_hier.py:210-224`) and `_combat_rewards` with mode "HighLevel"
For each tick, accumulated (undiscounted) into the commander-step reward:
* agent out of bounds: −2·s;
* agent kills opponent: **+1** (not scaled by `rew_scale`);
* agent killed by opponent: −1·s;
* friendly kill: no reward, but `alive_agents −= 1`.

If `glob_frac > 0` (default 0): `rewards[i] += own + glob_frac·Σ others`. Rewards are only added for agents that exist or are in `destroyed_ids`.

### 7.3 Evaluation (`evaluation.py`)
* 1000 episodes (`evaluation.py:19`). `eval_hl = True` uses the commander policy with `explore = False`.
* **[BUG]** The GRU states are reset to zeros **every commander step** and chained across agents within a step (`evaluation.py:41-48`). Commander memory is effectively unused at evaluation time, and agent k receives agent k−1's state.
* `eval_hl = False`: all agents get action 1 (nearest opponent) and the opponents use `fight_{ac}_opp`.
* The info metrics come from `env_base.py:91-107`:
  * `agents_win = alive_opps <= 0 and steps < horizon`;
  * `opps_win = alive_agents <= 0 and steps < horizon`;
  * `draw = steps >= horizon and both sides alive`;
  * fight and escape action counts, and `opp{k}` selection counts. Here `v` is the post-assessment commander action, so for opponents it is the attacked-agent index.

---

## 8. Rewards (LowLevel) (`env_base.py:240-310`, `env_hetero.py:188-225`)

`s = rew_scale` (default 1). Each step:

1. Inside `_take_action`: `rewards[i] = 0` for each agent alive at action time. The escape-mode fire penalties of −0.1 (§3) are added here.
   * `opp_stats[i] = [focus(opp_to_attack[i] → i)/180, dist_deg]` is recorded **before** the tick, if the target exists.
2. After `do_tick`, `_combat_rewards`:
   * **Out of bounds**, checked for ids 1..T in order (rockets are never checked): remove the unit; agents get `rews[i] += −5·s` and are added to `destroyed_ids` with `alive_agents −= 1`; opponents get `alive_opps −= 1` with no reward to anyone.
   * **Events**, in emission order:
     * agent killer, opponent victim, fight mode:
       * rocket kill (`origin.id >= T + 1`): `+ (1 + 0.5·missile_remain/rocket_max)·s`, i.e. range [1, 1.5]. `missile_remain` has already been decremented;
       * cannon kill: `+ [(0.5 + 0.5·cannon_remain/cannon_max) + (0.5 + 0.5·opp_stats[i][0])]·s`, i.e. range [1, 2]. `opp_stats` refers to the *pre-tick target*, even if the cannon killed a different opponent. `cannon_remain` is post-tick;
       * escape mode: 0;
       * `alive_opps −= 1`.
     * agent killer, agent victim (friendly fire): `rews[killer] += −2·s`. If `friendly_punish` (default False), the victim also gets −2·s and is added to `destroyed_ids`. `alive_agents −= 1`.
     * opponent killer, agent victim: `rews[victim] += −2·s`, add to `destroyed_ids`, `alive_agents −= 1`.
     * opponent killer, opponent victim: `alive_opps −= 1`.
3. `esc_dist_rew` (default False, escape only): for each living agent and each living opponent `j`, nearest first with rank `j = 1, 2, …`:
   * if `dist_deg < 0.06`: `−0.02/j`, plus another `−0.02/j` if speed < 200;
   * if `dist_deg > 0.13`: `+0.02/j`, plus another `+0.02/j` if speed > 500.
4. Sum: for agents that exist or are in `destroyed_ids`, `rewards[i] += Σ rews[i]`. If `glob_frac > 0` and fight mode: `+ glob_frac·Σ rews[i % 2 + 1]` (2v2 only).
   * **[BUG]** Kill rewards earned by an agent that died earlier in the episode (rocket still flying), or that was friendly-killed this tick without `friendly_punish`, are dropped.

Returned reward dict: keys are the agents alive at action time. Dead agents are absent. RLlib treats a missing reward as 0.0, see §10.4.

---

## 9. Termination (`env_base.py:79-109`)

* `terminated["__all__"] = truncated["__all__"] = (alive_agents <= 0 or alive_opps <= 0 or steps >= horizon)`. **[BUG]** `terminateds` and `truncateds` are *the same dict object*. There are no per-agent done flags.
* Because `terminated` is True at the horizon, RLlib uses `last_r = 0` (`postprocessing.py:181-182`). **The time limit is treated as a true terminal, with no bootstrapping.**
* Observations are returned for *all* agents 1..N every step, including dead ones (all-zero obs). Dead agents therefore keep producing transitions until the episode ends.
* `step({})` with an empty action dict does nothing and does not advance `steps`.

---

## 10. RLlib training setup

### 10.1 Common
* ray 2.4.0, torch, `framework("torch")`, `enable_connectors=False`, `batch_mode="complete_episodes"`, `num_rollout_workers = 4` (`config.py:34`), 1 env per worker, `num_gpus = 0` by default. The README reports that GPU was worse.
* `rollout_fragment_length`: the AlgorithmConfig default is 200 (`algorithm_config.py:304`; PPO does not override it in 2.4). With complete episodes, each worker returns ≥ 200 env steps of whole episodes. `synchronous_parallel_sample(max_env_steps = train_batch_size)` keeps sampling until ≥ `train_batch_size` env steps, so batches overshoot. **[AMBIG]** The exact batch size per iteration varies.
* `count_steps_by = "env_steps"`. In multi-agent, every env step yields one transition per agent, so ≈ 2000 per policy per iteration for hetero (2 policies × 1 agent each) and ≈ 3000 agent-steps per iteration for the commander (3 agents share one policy).
* Iterations: `epochs = 10000` per run (`config.py:35`), with `algo.train()` called `epochs + 1` times (`train_hetero.py:281-287`). There is no early stopping. A checkpoint plus FSP policy export happens every 50 iterations; render evaluation every 500.
* Warm start (`config.py:65-92`): fight L(n) restores the full algorithm state of L(n−1) if `results/L{n−1}_fight_2-vs-2` exists. Escape is trained first at L3 from scratch, then at L5 restored from L3 escape. The commander is trained from scratch.
  * **[BUG]** `type=bool` argparse flags turn any non-empty string, including `"False"`, into True.
* Curriculum procedure (README §Procedure): fight L1 → L2 → L3 → L4; escape L3; fight L5; escape L5 (optional); then the commander.

### 10.2 PPO hyper-parameters (RLlib 2.4 `PPOConfig` defaults at `ppo/ppo.py:__init__`, plus overrides)

| param | low-level (`train_hetero.py:216`) | commander (`train_hier.py:186`) |
|---|---|---|
| lr (Adam, torch defaults β = (0.9, 0.999), eps = 1e-8) | 1e-4 | 1e-4 |
| gamma | 0.99 | 0.99 (per commander step) |
| lambda (GAE) | 0.95 | **1.0 (default; not set)** |
| clip_param | 0.25 | 0.25 |
| kl_coeff (initial) | 0.2 | 0.2 |
| kl_target | 0.025 | 0.05 |
| vf_loss_coeff | 1.0 | 1.0 |
| vf_clip_param (clips the *squared* VF error) | 10.0 | 10.0 |
| entropy_coeff | 0.0 | 0.0 |
| num_sgd_iter | 30 | 30 |
| sgd_minibatch_size | 256 | 256 |
| train_batch_size (env steps) | 2000 | 1000 |
| grad_clip | None | None |
| model max_seq_len | 20 (catalog default) | 20 |
| exploration | StochasticSampling | StochasticSampling |

Loss (`ppo_torch_policy.py: loss`):

```
L = mean(−min(A·ρ, A·clip(ρ, 1 ± 0.25)) + 1.0·clip((V − V_targ)², 0, 10)) + kl_coeff·mean(KL(π_old ‖ π))
```

* Advantages are standardised per policy over the whole train batch (`ppo.py: training_step`, `standardize_fields`).
* The KL coefficient is adapted once per iteration from the mean KL of the last SGD epoch: if KL > 2·target, multiply by 1.5; if KL < 0.5·target, multiply by 0.5 (`torch_mixins.py:87-94`).
* Optimiser path: multi-agent with a `TorchPolicyV2` subclass makes `simple_optimizer = True` (`algorithm_config.py:909-945`), so `train_one_step` → `do_minibatch_sgd` runs **each policy separately**: 30 epochs × shuffled minibatches, one Adam step per minibatch (`utils/sgd.py`).
* Action distribution: `TorchMultiCategorical`, with logits split [13, 9, 2(, 2)], log-probabilities summed, entropy and KL summed. The commander uses `TorchCategorical(3)`.
* GAE: `δ_t = r_t + γV_{t+1} − V_t`, `A = discount_cumsum(δ, γλ)`, `V_targ = A + V_rollout`, with `V_rollout` taken from the rollout-time forward pass.

### 10.3 Low-level policies and centralised critic (`train_hetero.py:109-245`)
* Policies `ac1_policy` (agent 1, AC1) and `ac2_policy` (agent 2, AC2): `policy_mapping_fn = f"ac{agent_id}_policy"`.
* `observation_fn = central_critic_observer`. Each agent's obs is `Dict{obs_1_own: own obs, obs_2: partner obs, act_1_own: zeros, act_2: zeros}`.
* The gymnasium Dict sorts keys, and the RLlib Dict preprocessor flattens in that order: **`[act_1_own | act_2 | obs_1_own | obs_2]`**.
  * AC1 flat: [4 | 3 | 26 | 24] = 57 (fight) or [4 | 3 | 30 | 29] = 66 (escape).
  * AC2 flat: [3 | 4 | 24 | 26] or [3 | 4 | 29 | 30].
* `CustomCallback.on_postprocess_trajectory` (`train_hetero.py:120-160`) overwrites the flat columns 0–6 **in place** with normalised actions: `a0/12, a1/8, a2, (a3)`. The own action goes first, then the partner's action. Layout check: AC1 gets cols 0–3 = own and 4–6 = partner; AC2 gets 0–2 = own and 3–6 = partner (4 dims).
  * **[BUG] Critic train/rollout mismatch.** The callback runs **after** `postprocess_trajectory`, i.e. after GAE (`simple_list_collector.py:513-538`). `VF_PREDS`, and hence the advantages and value targets, are computed with **zero actions** in the critic input, while the critic is *trained* on inputs that contain the real actions.
  * **[BUG]** With an episode of length 1, `np.squeeze` makes the action array 1-D and the `act[:, 0]` indexing raises an exception.
* Rollout-time critic: the actor ignores the `act_*` inputs, so actions are unaffected.

### 10.4 Dead agents in the batch
RLlib 2.4 fills a missing reward with 0: `rewards[env_id].get(agent_id, 0.0)` (`sampler.py:884`, `env_runner_v2.py:542`). The env returns observations for all agents, so agents with zero obs after death contribute transitions (obs 0, reward 0, sampled actions, not terminal until `__all__`) to both GAE and the loss.

### 10.5 Models (`models/ac_models_hetero.py`, `models/ac_models_hier.py`)
`SlimFC(in, out, tanh, orthogonal_)` = `nn.Linear` with **orthogonal weight init (gain 1)**, **bias 0**, followed by tanh (`misc.py: SlimFC`). Outputs have no activation. `F.normalize` = L2 normalisation along dim 1, eps 1e-12.

**[BUG] One global `SHARED_LAYER` = SlimFC(500 → 500, tanh)** at module level (`ac_models_hetero.py:22-27`, `ac_models_hier.py:14-19`).
* Within a process, the *same* module object is used by **both the actor and the critic** of a model **and by both `ac1_policy` and `ac2_policy`**.
* Each policy's optimizer contains it. During `do_minibatch_sgd`, ac1's 30 epochs update it first, then ac2's 30 epochs; the two Adam states are separate.
* The exported FSP `.pt` files each hold a separate pickled copy.

**Fight1 (AC1, obs 26, logits 26)** (`ac_models_hetero.py:181-291`):
* actor:
  * `a = tanh(W1·o[:12])` (200); `b = tanh(W2·o[12:])` (14 → 200); `f = tanh(W3·o)` (26 → 100);
  * `att = MHA(100, heads = 2, batch_first)(F, F, F)`, where `F = add_time_dimension(f, seq_lens)` is `[B, T, 100]`;
  * `f' = normalize(f + att)`;
  * `logits = W_out·SHARED(concat[a, b, f'])` (500 → 26).
* critic:
  * `v1 = tanh(·)` on `[o_own(26), act_own(4)]` (30 → 175); `v2` on `[o_2(24), act_2(3)]` (27 → 175); `v3` on all 57 → 150;
  * `MHA(150, 2)` over time; `y' = normalize(v3 + att)`;
  * `V = W_v·SHARED(concat[v1, v2, y'])` (500 → 1).
* `get_initial_state = [zeros(1)]` (dummy). `forward` returns state `[]`.

**Fight2 (AC2, obs 24, logits 24)**: identical structure with own slice `o[:10]` (→ 200), `o[10:]` (14 → 200), `o` (24 → 100), v1 (27 → 175), v2 (30 → 175), v3 (57 → 150).

**[BUG / critical] Temporal attention behaves differently at rollout and at training time.**
* Because `get_initial_state()` is non-empty, the policy is treated as recurrent (`torch_policy_v2.py:173-174`). Missing `state_out_0` columns are filled from the initial state (`agent_collector.py:409`), so train batches are chopped into sequences of ≤ 20 steps per (episode, agent) and **zero-padded** to the max sequence length in the minibatch (`rnn_sequencing.pad_batch_to_sequences_of_same_size`, called in `torch_policy_v2.py:857`).
* Consequences:
  * At **rollout** (and FSP inference) `T = 1`, so the attention output is `out_proj(v_proj(f))`.
  * At **training**, the attention is **non-causal self-attention over the whole ≤ 20-step chunk, including future steps and padded rows** (no key padding mask). The policy being optimised is therefore not the policy that sampled the data: ρ ≠ 1 at the first SGD step.
  * Because `forward` returns an empty state list, the PPO loss takes the **non-RNN branch: plain `mean` over all rows including padding** (`ppo_torch_policy.py: if state:`). On padded rows the advantage is 0 (no policy-gradient term), `V_targ = 0` (the VF is regressed toward 0 there), `ACTION_DIST_INPUTS = 0` (the KL toward a uniform distribution is included), and the mean's denominator includes the padding.
  * Minibatches are formed from whole sequences (`sample_batch._get_slice_indices`), about 256 time steps each, with sequence order shuffled.

**Esc1 (AC1, obs 30)** (`ac_models_hetero.py:29-103`), plain feed-forward (`TorchModelV2`):
* actor: `concat[tanh(o[:7] → 150), tanh(o[7:25] → 250), tanh(o[25:] → 100)]` → SHARED → 26 logits;
* critic: `tanh(66 → 500)` on `[obs_1_own, act_1_own, obs_2, act_2]` → SHARED → 1;
* minibatches are fully shuffled rows.

**Esc2 (AC2, obs 29)**: actor `o[:6] → 150`, `o[6:24] → 250`, `o[24:] → 100` → SHARED → 24 logits; critic 66 → 500 → SHARED → 1.

**CommanderGru** (`ac_models_hier.py:21-112`), obs Dict flattened `[act_1_own(1) | act_2(1) | act_3(1) | obs_1_own(34) | obs_2(34) | obs_3(34)]`:
* actor:
  * `concat[tanh(o[:4] → 50), tanh(o[4:24] → 200), tanh(o[24:34] → 50)]`;
  * `f = tanh(o → 200)`; `g = GRU(200, 200)` over time with initial `h_act`; `f' = normalize(f + g)`;
  * → SHARED → 3 logits.
* critic:
  * `v1, v2, v3 = tanh(35 → 100)` on `[obs_k, act_k]` for own, other1, other2;
  * `v4 = tanh(105 → 200)` on all three; GRU(200, 200) with `h_val`; normalise the residual;
  * → SHARED → 1.
* State: `[h_act(200), h_val(200)]`, initialised to zeros. It returns a real state, so the PPO loss *is* masked. Sequences are ≤ 20 commander steps.
* Callback (`train_hier.py:105-132`): flat cols 0, 1, 2 = `action/2` for [own, lower-id other, higher-id other]. This matches the `central_critic_observer` ordering (`train_hier.py:134-165`). As in §10.3, the callback runs after GAE, so `VF_PREDS` were computed with zero actions. The callback hard-codes 3 agents.
* GRU / MHA initialisation: PyTorch defaults.
  * GRU: U(−1/√200, 1/√200).
  * MHA: `in_proj` xavier_uniform with bias 0; `out_proj` uses the default Linear init with bias 0.

### 10.6 FSP export
`algo.export_policy_model(dir, "ac{k}_policy")` writes `model.pt`, which is then renamed to `L{level}_AC{k}_{mode}.pt` (`train_hetero.py:101-105`, from level ≥ 3, every 50 iterations). The last export before a run ends is what later levels and the commander use. **[PORT]** Save a frozen parameter snapshot per level and mode instead.

---

## 11. Randomness inventory [NONDET]

| Draw | RNG | Where |
|---|---|---|
| side `r`, positions, headings, type of slots ≥ 2 | global `random` | `env_base.py:555-560`, `_sample_state` |
| L5 opponent policy k ∈ {3, 4, 5} | global `random` | `env_hetero.py:57` |
| cannon hit Bernoulli (0.15 / 0.30 per target per tick) | `sim.rnd_gen` (OS-entropy seeded every reset) | `ac1.py:112`, `ac2.py:99` |
| missile heading noise U(0.95, 1.05) per tick | global `random` | `ac1.py:127` |
| missile cooldown randint(7, 17) / randint(8, 12) | global `random` | `env_base.py:230` |
| scripted L1/L2/L3 coins, turns, speeds, escape timer, r ∈ U(0.7, 1.3) | global `random` | `env_hetero.py:118-271` |
| commander-opponent fight/escape and target choice | global `random.choices` / `randint` | `env_hier.py:175-186` |
| PPO action sampling, minibatch shuffles, init | torch / numpy / `random` | RLlib |

Bit-exact RNG parity is impossible. Aim for **distributional parity**, with one JAX key split per draw site.

---

## 12. Top risks for a faithful JAX port (summary)

1. **Training-time temporal attention in Fight1/Fight2** (§10.5): non-causal MHA over ≤ 20-step padded chunks at training versus T = 1 at rollout, plus an unmasked loss that includes padded rows (VF → 0, KL to uniform). Reproducing it requires RLlib-style sequence chunking. Removing it is a deliberate deviation.
2. **Critic inputs differ between GAE (zero actions) and training (real actions)** (§10.3, §10.5). There is also a `SHARED_LAYER` shared across the actor, the critic and both policies, with separate Adam states, updated sequentially per policy.
3. **Sequential, order-dependent tick semantics** (§2.3–2.5): id-ordered updates; units that die mid-tick still act; the rocket hit check is done before the rocket moves; the cannon hits immediately; the rocket proximity fuse on unit id 2.
4. **Weapon quirks that dominate the gameplay**: unguided missiles with multiplicative heading noise; the radar cone offset to [h, h + 120]; the cooldown is set on failed launches; scripted opponents never decrement `missile_wait`, so they get one missile attempt per episode.
5. **Environment-to-RL interface**: the horizon is treated as terminal (no bootstrap); dead agents keep emitting zero-obs/zero-reward transitions; batches overshoot `train_batch_size`; λ = 1.0 for the commander; the commander γ is per variable-length macro-step.
6. **HighLevel specifics**: frozen distances in low-level observations during a macro-step; the escape missile targets the farthest opponent; opponents start at speed 0 (`level` defaults to 1); opponents attack a non-nearest agent with prob 0.75 when one is available; the macro-step length (≤ 16 ticks, early exit on any kill or after tick 12 on a close engagement) can push episodes past 500 ticks.
7. **Geodesics and precision**: use the local ellipsoidal approximation (§1). Heading normalisation divides by 359. Observation features mix planar-degree geometry with geodesic weapon geometry, and both must be kept.
8. **Ammunition depends on level and mode** (§4.3; at L5 even the agents change), and the observation normalisers change with it.
