# Paper results and reproduction targets: HHMARL 2D (arXiv 2309.11247v1)

Source: Selmonaj, Szehr, Del Rio, Antonucci, Schneider, Rüegsegger, "Hierarchical Multi-Agent
Reinforcement Learning for Air Combat Maneuvering", arXiv 2309.11247v1 (20 Sep 2023), published at
ICMLA 2023. The paper is 8 pages and has **no numeric tables**. Every quantitative result is in a
figure (Figs. 6, 7, 8, 10). The values below were read from the PDF rendered at 300 dpi. Bar charts
were measured pixel by pixel, so treat them as about ±1 pp or ±5 counts. Learning curves were read
by eye, so treat them as about ±0.1 to 0.2 reward units.

Reference code: this repo (`IDSIA/hhmarl_2D`, HEAD `d5a23f2`; the paper-era commit is `5f38309`,
2023-09-22). Per `CLAUDE.md`, **follow the code by default** when it disagrees with the paper. Section 4
lists the disagreements.

---

## 1. Experimental setup as stated in the paper

### 1.1 Simulator / environment
- A custom 2D Python simulator. Altitude is constant. Within-visual-range dogfight. One simulation
  step is t (the code uses `tick_secs = 1` s).
- Map size per axis: **30 km for low-level training** and **50 km for high-level training**.
- Episode end: the time horizon is reached, or every aircraft of one team is destroyed.
- An aircraft is destroyed when it is hit by cannon or rocket, or when it leaves the map boundary.
- Each episode: each team gets a random side of the map (left or right). Initial positions and
  headings are random.
- Team composition: aircraft types are chosen at random, with **at least one of each type per
  team**.
- Outcome definitions (evaluation):
  - **Win**: all opponents destroyed.
  - **Loss**: all agents destroyed.
  - **Draw**: at least one aircraft per team is still alive at the end of the episode.
- Evaluations use **1,000 episodes** each.

### 1.2 Aircraft types (heterogeneous, based on the Dassault Rafale)
| Property | AC1 (original Rafale) | AC2 (modified) |
|---|---|---|
| Angular velocity [deg/s] | [0, 5] | [0, 3.5] |
| Speed [knots] | [100, 900] | [100, 600] |
| Cannon WEZ angle [deg] | [0, 10] | [0, 7] |
| Cannon WEZ range [km] | [0, 2] | [0, 4.5] |
| Hit probability | 0.70 | 0.85 |
| Rockets | yes | no |

AC1 is more agile and carries rockets. AC2 has a longer cannon range and no rockets.

### 1.3 Policies and hierarchy
- Low-level (options): fight policy π_f and escape policy π_e, **one instance per aircraft type**. That
  gives 4 low-level policies [π_f,AC1, π_f,AC2, π_e,AC1, π_e,AC2]. All agents of the same type share
  parameters.
- High-level: one commander policy π_h shared by all agents and both aircraft types. It is called
  per agent.
- The training scheme is CTDE. The critic sees the observations and actions of all interacting
  agents. The algorithm is PPO (actor-critic).
- Training runs in two stages: (1) train the low-level policies, then (2) freeze them and train π_h
  with the low-level policies treated as part of the environment.

### 1.4 Observations (all normalized to [0, 1])
Metrics: heading α_h, heading-off α_off, aspect angle α_AA, antenna-train angle α_ATA, distance d,
position (x, y), speed s, cannon ammo c1, rockets c2 (AC1 only), rocket-ready indicator w (AC1),
currently-shooting indicator s_r.

- **π_f** observes the closest opponent and the closest friendly aircraft:
  - o_a = [x, y, s, α_h, α_off,o, α_AA,o, α_ATA,o, d_o, c1, (c2, w, s_r)_AC1]
  - o_o = [x, y, s, α_h, α_off,a, α_AA,a, α_ATA,a, d_a, s_r]
  - o_fr = [x, y, s, α_off,a, α_ATA,a, α_ATA,fr, d_a, s_r]
  - o_full = o_a ‖ o_o ‖ o_fr
- **π_e** observes the 2 closest opponents and the closest friendly aircraft:
  - o_a = [x, y, s, α_h, c1, (c2)_AC1]
  - o_o = [x, y, s, α_h, α_off,a, α_AA,a, α_ATA,a, α_ATA,o, d_a]
  - o_fr = [x, y, s, α_h, α_ATA,a, α_ATA,fr, d_a]
  - o_full = o_a ‖ o_o1 ‖ o_o2 ‖ o_fr
- **π_h** observes the 3 closest opponents and the 2 closest friendly aircraft:
  - o_a = [x, y, s, α_h]
  - o_o = [x, y, s, α_h, α_AA,a, α_AA,o, α_ATA,a, α_ATA,o, d_a]
  - o_fr = [x, y, s, d_a]
  - o_full = o_a ‖ o_o1 ‖ o_o2 ‖ o_o3 ‖ o_fr1 ‖ o_fr2

### 1.5 Actions (discrete)
- Low level (π_f and π_e use the same action set):
  - relative heading h ∈ {−6..6}, giving α_h ← −15·h + α_h (turn range [−90°, 90°]);
  - velocity v ∈ {0..8}, mapped onto the speed range of AC1 or AC2;
  - cannon c ∈ {0, 1};
  - rocket r ∈ {0, 1} (AC1 only).
- Commander: a_c ∈ {0, 1, 2, 3}. 0 activates π_e. 1, 2, 3 activate π_f against the 1st, 2nd or 3rd
  observed opponent.

### 1.6 Rewards
- **π_f**:
  - On a kill: r_k = α_ATA,a + (c_max − c_rem)/c_max ∈ [1, 2]. Here c_rem = c1 + c2, and α_ATA is
    "of the opponent to the agent".
  - Out of bounds: r_b = −5.
  - Destroying a friendly aircraft: r_f = −2.
  - **No per-step reward.** r = r_k + r_b + r_f.
- **π_e**:
  - Per step: r_e = −0.01 if d < 6 km, +0.01 if d > 13 km, and 0 otherwise. The distance is to the
    opponents.
  - r = r_e + r_b + r_f.
- **π_h**:
  - r_f = +1 if the agent (running its low-level policy) killed an opponent, and −1 if the agent was
    killed.
  - Situation reward: r_c = +0.1 if d_o < 5 km ∧ α_ATA,o < 30° ∧ α_AA,o < 50° ∧ a_c > 0, and 0
    otherwise.
  - Out of bounds: r_b = −5.
  - r = r_f + r_c + r_b.

### 1.7 Curriculum (low-level, fictitious self-play)
- Five levels with increasingly capable opponents:
  - **L1**: static.
  - **L2**: random.
  - **L3**: scripted. The opponent engages the closest agent with α_ATA ≈ 0 and escapes at random.
  - **L4**: opponents use the L3 policy.
  - **L5**: opponents use the L1–L4 policies, chosen at random each episode.
- After a level finishes, the policy is transferred to the next level and training continues.
- Horizon: **T = 200 at L1, +50 per level**, so L1 200, L2 250, L3 300, L4 350, L5 400.
- Ammunition per episode:
  - agents: **200 cannon "shots"** and **5 rockets** (AC1);
  - opponents: **400 cannon and 8 rockets** (made stronger).
- Low-level training is **2 vs 2**. The fight curriculum runs L1 to L5.
- **π_e is trained only at L3** (Fig. 8a).
- Commander training is **3 vs 3**. Agents and opponents use π_f of L5 or π_e. Opponents mostly fight
  and escape at random. Ammunition is 300 cannon and 8 rockets.
- Commander rollout (Alg. 2):
  - high-level horizon T_h = 40, low-level horizon T = 10 per commander decision;
  - each episode samples a 3v3 scenario;
  - the low-level loop breaks when t ≥ 10 or an event occurs. Fig. 2 instead says "call on event or
    every 20 low level steps".
  - Events:
    - any aircraft destroyed (shot, or hit the boundary);
    - an agent approaches the boundary (d < 6 km);
    - an agent or opponent is in the favorable situation of Eq. 3;
    - two opponents are close and face an agent (d < 5 km, α_ATA,a < 30°).

### 1.8 Network architecture (Fig. 5)
- **Actor**:
  - o_a → FC-tanh 200;
  - o_o‖o_fr → FC-tanh 200;
  - o_full → embedding (linear, 100, tanh) → SA (π_f) or GRU (π_h) with a skip connection →
    Add&Norm (100);
  - concat (500) → **shared FC-tanh 500** → FC-tanh 500 → π.
- **Critic**:
  - O_t → FC-tanh 300;
  - A_t → FC-tanh 100;
  - O_t‖A_t → embedding → SA/GRU → Add&Norm (100);
  - concat (500) → shared FC-tanh 500 → FC-tanh 500 → Q/V.
- The green **shared layer** is shared between AC1 and AC2 networks, and between actor and critic.
- π_e uses neither SA nor GRU.
- The commander has one network instance for both aircraft types.
- Baseline "FC": a fully connected network with **2 layers of 500 neurons, tanh**.

### 1.9 PPO hyperparameters (paper, Sect. IV-B)
| Param | Value |
|---|---|
| learning rate (actor and critic) | 1e-4 |
| discount γ | 0.95 |
| clip ε | 0.2 |
| optimizer | Adam |
| batch size | 2,000 (low level), 1,000 (high level) |
| update iterations K | not stated |
| GAE λ, minibatch, entropy, value coefficient, gradient clip | **not stated** |

Libraries: Ray RLlib and PyTorch. The paper reports a "standard RL system" baseline that "showed
very poor performance", but it is not shown.

### 1.10 Training budget, hardware, seeds
- **Not stated in the paper**: wall-clock time, hardware, number of seeds, number of iterations per
  level. The only budget information is the x-axes of the learning curves:
  - Fig. 6, π_f at L3: about 28M;
  - Fig. 8a, π_e at L3: about 8.2M;
  - Fig. 10a, π_h: about 6.7M.
- The axes are labelled "Iterations" but their units are millions. This matches RLlib's TensorBoard
  x-axis `timesteps_total` (env steps sampled), not training iterations. The code defaults to
  `epochs = 10,000` iterations. With `train_batch_size = 2000` and
  `batch_mode = "complete_episodes"` (which overshoots by part of an episode per worker), 10k
  iterations come to roughly 20M–28M env steps. That is consistent with Fig. 6.
- Seeds: curves look like **single runs**, shown as a TensorBoard-smoothed line over a faint raw
  curve. There is no seed variance and there are no error bars.
- Hardware, from the README rather than the paper: RTX 3080Ti and i9-13900H. CPU training was
  reported faster or better than GPU in RLlib.

---

## 2. Quantitative results

There are no tables. Each figure is described below.

### Fig. 6: π_f training performance at L3 (architecture and curriculum ablation)
- x: "Iterations", 0 to 28M (env steps). y: mean episode reward summed over all agents, range about
  −2 to 1.5. Gridlines are every 0.5 and there is a bold line at 0.

| Curve | Start | Notable points | Final (~27–28M) |
|---|---|---|---|
| **SA-Net** (with curriculum) | ≈ −2.0 at ~1M | ≈ −1.3 at 8M; ≈ −0.7 at 14M; crosses 0 at ~19–20M; peak ≈ 1.0–1.1 at ~25–26M | **≈ 0.95** |
| **SA-Net (no Curr)** | ≈ −2.0 at ~1M | fast rise to ≈ 0 by ~11–12M; plateau ≈ 0 to 0.2 between 12M and 20M; then rises | **≈ 0.6** |
| **FC** (2×500 tanh) | ≈ −1.8 at ~4M (curve starts late) | ≈ −1.0 at 5M; ≈ −0.7 at 12M; ≈ −0.5 at 20M | **≈ 0.0 to 0.05** |

Takeaways:
- Final ordering: SA-Net > SA-Net (no Curr) > FC.
- The curriculum run is slower early but ends higher. SA-Net overtakes no-Curr at about 21–22M.
- No curve has fully plateaued by 28M.

### Fig. 7a: "Evaluation Level 5", π_f 2v2 statistics (counts over 1,000 episodes)
Setup: agents use π_f of L5 and opponents use π_f of L4. Labels: k = kills of an opponent, d = agent
destroyed, fk = friendly kill. The suffix is the aircraft type. The y-axis runs 0–800.

| k-1 | k-2 | d-1 | d-2 | fk-1 | fk-2 | Draw |
|---|---|---|---|---|---|---|
| **≈ 730** | **≈ 232** | **≈ 196** | **≈ 481** | **≈ 50** | **≈ 39** | **≈ 162** |

Notes:
- AC1 kills about 3× as often as AC2.
- AC2 is destroyed about 2.5× as often as AC1.
- There are ≈ 962 opponent kills against ≈ 677 own losses in 1,000 episodes, with a maximum of 2,000
  possible kills.
- Draw ≈ 16%. Fig. 7b shows 18.5% draws for 2v2, probably from a different evaluation run.

### Fig. 7b: "Evaluation Fight", π_f (L5) vs opponents π_f (L4), n-vs-n (trained in 2v2)
| Scenario | Win | Loss | Draw |
|---|---|---|---|
| 2vs2 | **71.4%** | 10.1% | 18.5% |
| 3vs3 | **65.3%** | 10.9% | 23.8% |
| 4vs4 | **56.3%** | 13.9% | 29.8% |
| 5vs5 | **47.3%** | 16.1% | 36.6% |

Trend: as team size grows, wins fall and draws rise. Win/loss stays strongly positive (about 3:1 or
better).

### Fig. 8a: π_e training performance at L3
- x: "Iterations", 0 to about 8.2M. y: mean reward, about −3 to 7.
- Start ≈ −3 at ~0.2M; crosses 0 at ~1.1–1.2M.
- Plateau ≈ +1 between 1.5M and 2.8M (a dip to ≈ −0.5 at about 2.7M).
- Rises to ≈ 3–4 between 3.5M and 5M, and reaches ≈ 5 by about 5.8M.
- Plateau ≈ 4.5–5.5 between 6M and 8.2M (peaks ≈ 6.5). **Final ≈ 5.3**.

### Fig. 8b: "Evaluation Escape", π_e in n-vs-n
Categories:
- **Escaped**: no agent killed, including out-of-bounds.
- **Killed**: at least one agent killed.
- **Kills**: at least one opponent killed.

| Scenario | Escaped | Killed | Kills |
|---|---|---|---|
| 2vs2 | **79.4%** | 15.9% | 4.7% |
| 3vs3 | **58.2%** | 33.1% | 8.7% |
| 4vs4 | **31.1%** | 52.2% | 16.7% |
| 5vs5 | **14.2%** | 67.2% | 18.7% |

The opponents are not stated. They are presumably fight policies (L5 π_f in the code).

### Fig. 10a: π_h training performance (3v3), architecture comparison
- x: "Iterations", 0 to about 6.7M. y: mean reward, about −3.5 to 3.5.

| Curve | Start | Saturation | Final |
|---|---|---|---|
| **GRU-Net** (chosen) | ≈ −3.3 | ≈ 1.0 by ~1.0–1.2M, then flat about 0.8–1.1 | **≈ 0.8–0.9** |
| **SA-Net** | ≈ −3 | ≈ 0 by ~1.5M; flat about −0.1 to 0.3 | **≈ 0.2** |
| **FC** | ≈ −2.6 | ≈ −0.5 by ~1.5M; flat about −0.5 to −0.2 | **≈ −0.2** |

All models saturate quickly, at about 1–1.5M. GRU sits about 0.7 to 1.0 above SA, and SA sits about
0.3 to 0.4 above FC.

### Fig. 10b: "Evaluation Commander", π_h with L5 low-level policies, n-vs-n
| Scenario | Win | Loss | Draw |
|---|---|---|---|
| 2vs2 | **51.2%** | 29.1% | 19.7% |
| 3vs3 | **46.0%** | 28.1% | 25.9% |
| 4vs4 | **35.1%** | 26.1% | 38.8% |
| 5vs5 | **23.1%** | 20.9% | 56.0% |

Paper's interpretation:
- The commander helps in small teams.
- In larger teams, results tend toward draws with win ≈ loss, which is what you would expect with no
  commander, since both teams are on the same level.
- Possible causes the paper names: sensing only 3 opponents, the stochasticity of opponents switching
  between fight and escape, and the per-team AC1/AC2 mix.

### Fig. 9
Qualitative trajectories only: commander 2v4, fight, escape. There are no numbers.

---

## 3. Reproduction targets (checklist)

Tolerances assume 1,000 evaluation episodes (binomial standard error ≈ 1.5 pp) plus unknown seed
variance, since the paper reports a single run. Suggested tolerances: **±7 pp** on win, loss and draw
rates, and **±0.3** on smoothed final reward for π_f and π_h. Escape rewards are about 5× larger in
scale, so use ±1.0 for π_e. Qualitative orderings and trends are **hard requirements**. Absolute
values are soft, because the reward code differs from the paper (Section 4). Measure curves the same
way the paper does: mean episode return summed over the team's agents, plotted against env steps.

**Low-level fight (π_f)**
- [ ] Fig. 6: at about 28M env steps of L3, final reward is SA-Net ≈ **1.0 (±0.3)**, SA-Net
      no-curriculum ≈ **0.6 (±0.3)**, FC ≈ **0.0 (±0.3)**. Ordering SA > SA-noCurr > FC must hold.
      SA-Net should cross 0 by about 20M (±5M).
- [ ] Fig. 7b, L5 π_f vs L4 π_f:
  - 2v2 win **71% ±7**, loss **10% ±5**, draw **19% ±7**;
  - 3v3 win **65% ±7**;
  - 4v4 win **56% ±7**;
  - 5v5 win **47% ±7**.
  - Win must decrease and draw must increase monotonically with team size. Loss must stay at or
    below about 16%.
- [ ] Fig. 7a, 2v2 L5 vs L4 counts per 1,000 episodes:
  - k-1 ≈ **730 ±70**, k-2 ≈ **230 ±50** (AC1 kills / AC2 kills ≥ 2.5);
  - d-1 ≈ **200 ±50**, d-2 ≈ **480 ±70** (AC2 destroyed more than AC1);
  - friendly kills about 40–50 each (fewer than 100);
  - draws ≈ **160 ±50**.

**Low-level escape (π_e, L3 only)**
- [ ] Fig. 8a: reward goes from about −3 to a plateau of **≈ 5 (±1.0)** after about 6M env steps,
      and crosses 0 by about 1.2M.
- [ ] Fig. 8b, escaped rate: 2v2 **79% ±7**, 3v3 **58% ±7**, 4v4 **31% ±7**, 5v5 **14% ±7**.
      Kills: 2v2 ≈ **5%**, rising to about **19%** at 5v5. Escaped must decrease monotonically with
      team size.

**Commander (π_h, 3v3 training)**
- [ ] Fig. 10a: saturation by about 1–1.5M env steps. Final GRU ≈ **0.9 (±0.3)**, SA ≈ **0.2
      (±0.3)**, FC ≈ **−0.2 (±0.3)**. Ordering GRU > SA > FC must hold.
- [ ] Fig. 10b:
  - 2v2 win **51% ±7**, loss **29% ±7**;
  - 3v3 win **46% ±7**, loss **28% ±7**;
  - 4v4 win **35% ±7**, draw **39% ±7**;
  - 5v5 win **23% ±7**, draw **56% ±7**.
  - Win must exceed loss at 2v2 and 3v3. Draw must rise monotonically with team size.

**Budget guidance (inferred, not stated)**
- [ ] About 10k PPO iterations per curriculum level. That is about 20–28M env steps for L3 fight,
      about 8M for escape and about 6–7M commander env steps. Report wall-clock time and hardware,
      since the paper gives none. Run at least 3 seeds and report mean ± std, since the paper gives
      a single run.

---

## 4. Discrepancies: paper vs code (current HEAD unless stated)

Policy (`CLAUDE.md`): implement what the code does. The paper value is listed for reference.

### Simulation and dynamics
| Item | Paper | Code |
|---|---|---|
| Cannon hit probability | AC1 0.70, AC2 0.85 | `ac1.py` 0.75, `ac2.py` 0.90, both in HEAD and the paper-era commit. Applied per 1 s tick as p_hit/burst_secs while the target is in the cone: 0.15/s for AC1 (5 s burst), 0.30/s for AC2 (3 s burst). |
| WEZ angle | [0, 10]°, [0, 7]° | `cannon_width_deg` 10 / 7 is the full cone width. The target must be within ±5° / ±3.5° of heading. |
| Ammunition units | "200 cannon shots" | `cannon_max_time_sec = 200` is **seconds of burst**. One trigger gives a 5 s (AC1) or 3 s (AC2) burst, so about 40 / 67 bursts. |
| Heading action | α_h ← −15·h + α_h | `(a−6)*15` added to heading (sign flipped; a ∈ 0..12). |
| Map size | 30 / 50 km | `map_size` 0.3 / 0.5 **degrees** of lat/lon around (lat 5°, lon 7°). That is about 33 / 55 km. Distances in rules are in degrees: 0.1° ≈ 11 km, 0.06° ≈ 6.6 km, 0.13° ≈ 14.4 km. |
| Rocket mechanics | not described | Rocket kills if within 1 km of its target. Lifetime 10 s, speed profile 500→2000 kn. Heading is not guided: it is jittered ×U(0.95, 1.05) each tick. **Bug**: firing requires bearing ∈ [heading, heading+120°] (the cone is offset by +60°, see `_angle_in_radar_range`). **Bug**: the rocket's friendly-fire check hard-codes unit IDs 1/2. |
| Rocket cooldown | indicator w | `missile_wait` = randint(7,17) steps (low level) or randint(8,12) (high level). It counts down only when no rocket is in flight. |

### Curriculum and episodes
| Item | Paper | Code |
|---|---|---|
| Horizon per level | 200, 250, 300, 350, 400 | `config.py`: **{1:150, 2:200, 3:300, 4:350, 5:400}**. The paper-era commit used {150, 200, 300, 300, 400}. |
| L5 opponents | policies L1–4, random | `env_hetero.reset`: k ∈ {3, 4, 5} picks the **L3 fight, L4 fight or L3 escape** policy. L1 and L2 are never used. |
| L1 / L2 opponents | static / random | L1: speed 0, but AC1 opponents still fire rockets at random. L2: random ±90° turns, speed 100 + 75·k, **cannon fired every step**, random rockets. Both start at speed 0 for L≤2. |
| L3 script | engage with α_ATA ≈ 0, escape at random | Every 60 steps there is a 50% chance that **all** opponents escape diagonally for 20–30 steps. |
| Opponent ammo | 400 cannon, 8 rockets | Matches for L1–L4. At **L5, both teams** get 300 cannon and 6 rockets (not stated in paper). High level: 300 / 8 for both teams (matches). |
| Escape training | L3 only | Code trains π_e at L3 (vs the scripted opponent) and then optionally at L5 against L5 π_f (README step 4). The HL env loads `L5_*_escape.pt` if present, otherwise L3. |
| Initial spawn | random side, position, heading | Per-level spawn boxes in `_sample_state` (tight at L1 with constrained headings). Opponents at L1 have heading 0. Agent unit 1 is AC1, unit 2 is AC2, units 3+ are random. |

### Rewards
| Item | Paper | Code |
|---|---|---|
| π_f kill reward | α_ATA + (c_max − c_rem)/c_max ∈ [1, 2] | Cannon kill: `shift(cannon_remain/cannon_max → [0.5, 1]) + shift(ATA_norm(opp→agent) → [0.5, 1])` ∈ [1, 2]. This rewards **remaining** ammo. Rocket kill: `shift(missile_remain/rocket_max → [1, 1.5])`. |
| π_f being killed | not mentioned | **−2** when shot down by an opponent. |
| Friendly kill | −2 | −2 to the killer. The victim also gets −2 only if `friendly_punish` (default False). |
| Out of bounds | −5 | −5 at low level. **−2 at high level** (paper says −5). |
| π_e per-step reward | ±0.01 at <6 km / >13 km | `esc_dist_rew` **defaults to False**, so there is no distance reward. When enabled: ±0.02/j per opponent j (sorted), plus another ±0.02/j if speed < 200 or > 500 kn, with thresholds 0.06° and 0.13°. |
| π_e extra | none | −0.1 for firing the cannon when cannon_remain < 90 s, and −0.1 for firing a rocket when < 3 left. **No kill reward** in escape mode. Being killed gives −2. |
| π_h situation reward | +0.1 if d < 5 km ∧ ATA_o < 30° ∧ AA_o < 50° ∧ a_c > 0 | Fight action: +0.1 if dist < 0.1° (~11 km) ∧ ATA(agent→opp) < 15° ∧ ATA(opp→agent) > 40°. Escape action: +0.1 if the closest opponent has ATA(opp→agent) < 15° ∧ ATA(agent→opp) > 40° within 0.1°. Selecting a non-existent opponent gives **−0.1**. The paper-era code used ±0.5 with 30° thresholds. |
| π_h kill / killed | +1 / −1 | Matches. |
| Reward sharing | – | `glob_frac = 0` (none). `rew_scale = 1`. |

### Commander (high level)
| Item | Paper | Code |
|---|---|---|
| Sensed opponents / actions | 3 opponents, a ∈ {0..3} | HEAD: **`N_OPP_HL = 2`**, `Discrete(3)` (escape, opp1, opp2). The paper-era commit used `Discrete(4)` with obs_dim 41. |
| Observation | 4 + 3×9 + 2×4 = 39 | HEAD: 4 + N_OPP_HL×10 + 2×5 = **34**. Per opponent: x, y, s, h, heading-diff, ATA×2, AA×2, d. Per friend: x, y, ATA×2, d (no speed). |
| Invocation | T_h = 40, T = 10, events (Alg. 2) / "every 20 steps" (Fig. 2) | HEAD: up to 16 low-level steps (`while s <= 15`). It stops early on any kill or boundary event. The situation event (dist < 0.1° and ATA < 15° either way) is only checked after 11 steps. Episode horizon is **500 low-level steps** with no T_h cap, so about 31–50 commander steps. The paper-era commit used sub_steps 20 and horizon 400. |
| Events | boundary approach < 6 km, 2 opponents < 5 km | Not implemented in HEAD (the paper-era code had a boundary check at 15%). |
| Opponent behaviour | "mainly fight, randomly escape" | `hier_opp_fight_ratio = 75`% fight. When fighting with more than one candidate, a **75% chance to target a random non-closest agent**. |

### Observations (low level)
- π_f obs dims in code: **AC1 26, AC2 24**.
  - Own: x, y, s, h, ATA, AA, heading-diff, d, cannon, (rockets, ready, shooting)_AC1 or shooting_AC2.
    That is 12 values for AC1 and 10 for AC2.
  - Opponent (9): x, y, s, h, heading-diff, ATA(opp→ag), AA, d, shooting.
  - Friend (5): x, y, ATA, ATA, d. The paper's friend vector has 8 entries, including speed,
    heading-off and shooting.
- π_e obs dims: **AC1 30, AC2 29**: own 7 or 6, plus 2 opponents × 9, plus friend 5.
- Heading is normalized as (h mod 359)/359. Distance is normalized by √2·map_size.

### Networks
- π_f actor (`Fight1`/`Fight2`):
  - FC-tanh 200 (own 12 or 10 features) and FC-tanh 200 (rest);
  - FC-tanh 100 (full obs) → `nn.MultiheadAttention(100, 2 heads)`, plus a skip connection, then
    **L2-normalize** (not LayerNorm);
  - concat 500 → shared FC-tanh 500 → **linear** logits. There is no extra FC-tanh 500 layer as in
    Fig. 5.
- π_f critic: FC-tanh 175 (own obs+act), FC-tanh 175 (friend obs+act), FC-tanh 150 (all) → MHA(150, 2
  heads) + skip + normalize → concat 500 → shared 500 → linear 1. The paper says 300 / 100 / 100.
- **SA operates over the time axis** of RLlib's RNN sequences (`add_time_dimension`, default
  max_seq_len 20) **with no causal mask**. During training it attends to future steps in the chunk.
  During rollout each step is a length-1 sequence. This is a train/inference mismatch that needs to
  be replicated or justified.
- `SHARED_LAYER` is a module-level global (500→500 tanh). It is shared across AC1/AC2, actor/critic
  and fight/escape models in the same process.
- π_e (`Esc1`/`Esc2`): FC 150 (own) + FC 250 (2 opponents) + FC 100 (friend) → shared 500 → linear.
  Critic: FC 500 on the concatenated obs and actions → shared 500 → linear.
- π_h (`CommanderGru`):
  - actor: FC 50 (own) + FC 200 (opponents) + FC 50 (friends); FC 200 (full) → GRU(200) + skip +
    L2-normalize; concat 500 → shared 500 → linear;
  - critic: 3×FC 100 + FC 200 → GRU(200) → shared 500;
  - separate actor and critic hidden states.
- The centralized critic input is built through an RLlib `observation_fn` that sets action slots to
  zeros. `on_postprocess_trajectory` then writes the real own and friend actions into the flattened
  obs before training, scaled as heading/12 and speed/8, with fire flags raw. The flattened Dict
  order is lexicographic: `act_1_own`, `act_2`, `obs_1_own`, `obs_2`. At rollout time the critic
  therefore sees zero actions. It sees real actions only in the training batch.

### PPO hyperparameters
| Param | Paper | `train_hetero.py` (low level) | `train_hier.py` (commander) |
|---|---|---|---|
| lr | 1e-4 | 1e-4 | 1e-4 |
| γ | **0.95** | **0.99** | **0.99** |
| clip ε | **0.2** | **0.25** | **0.25** |
| GAE λ | – | 0.95 | RLlib default **1.0** |
| train batch | 2000 / 1000 | 2000 | 1000 |
| minibatch | – | 256 | 256 |
| SGD epochs | – | RLlib default 30 | 30 |
| KL | – | kl_coeff 0.2 (default), kl_target 0.025 | kl_target 0.05 |
| entropy / vf_coeff / vf_clip / grad_clip | – | 0 / 1.0 / 10.0 / None (RLlib 2.4 defaults) | same |
| batch_mode | – | complete_episodes, 4 rollout workers | same |
| iterations | – | `epochs = 10000` per level; checkpoint every 50 | 10000 |

### Evaluation
- `evaluation.py` runs **1,000 episodes in `HighLevelEnv`** (50 km map, horizon 500) for both commander
  and low-level evaluation. It does not use the 30 km low-level env.
- Defaults:
  - commander evaluation: both teams use L5 π_f;
  - low-level evaluation (`eval_hl=False`): agents L5 vs opponents L4; agents always attack the
    closest opponent. Set `hier_opp_fight_ratio=100` for a pure fight evaluation.
- The 2v2–5v5 sweeps are run by changing `num_agents` / `num_opps`.
- Win/loss count only before the horizon. Draw means the horizon was reached with both teams alive.
- The per-type kill counts of Fig. 7a (k-1, d-2, fk-1, ...) are **not computed** by the released
  `evaluation.py`.
- There is no escape-evaluation script for Fig. 8b, and its opponent setup is unspecified.
