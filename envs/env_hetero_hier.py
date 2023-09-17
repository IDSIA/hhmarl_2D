"""
Environment for MARL Warsim, where 2 groups of airplanes fight against each other. 
"""

import torch
import time
import os
import random
import json
import numpy as np
from math import sin, cos, acos, pi, hypot, radians, exp, sqrt
from gymnasium import spaces
from pathlib import Path
from typing import List, Dict, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from warsim.scenplotter.scenario_plotter import PlotConfig, ColorRGBA, StatusMessage, TopLeftMessage, \
    Airplane, PolyLine, Drawable, Waypoint, Missile, ScenarioPlotter

from warsim.simulator.cmano_simulator import Position, CmanoSimulator, UnitDestroyedEvent
from warsim.simulator.rafale_rocket import Rafale
from warsim.simulator.rafale_long import RafaleLong
from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits
from utils.angles import sum_angles

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from models.ac_models_cc import CCRocket, CCRocketFri, CCRocketDummy
from models.ac_models_hetero import CCRocketDummyAC1, CCRocketDummyAC2,CCRocketAC1, CCRocketAC2, CCAtt1, CCAtt2, CCFc1, CCFc2, CCEsc1, CCEsc2, CCDummyEsc1, CCDummyEsc2
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.rllib.algorithms.algorithm import Algorithm

colors = {
    'red_outline': ColorRGBA(0.8, 0.2, 0.2, 1),
    'red_fill': ColorRGBA(0.8, 0.2, 0.2, 0.2),
    'blue_outline': ColorRGBA(0.3, 0.6, 0.9, 1),
    'blue_fill': ColorRGBA(0.3, 0.6, 0.9, 0.2),
    'waypoint_outline': ColorRGBA(0.8, 0.8, 0.2, 1),
    'waypoint_fill': ColorRGBA(0.8, 0.8, 0.2, 0.2)
}
knots_to_ms = 0.514444 # used for predicted state

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3

class BasicMarl(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True
        self.obs_dim = env_config.get("obs_dim", 0)
        self.curr_level = env_config.get("curr_level", 5)
        self.mode = env_config.get("mode", "fight")

        if self.curr_level == 5:
            self.obs_dim_map = {1: 30, 2:28, 3:30, 4:28} if self.mode == "fight" else {1: 30, 2:29, 3:30, 4:29}
            self._obs_space_in_preferred_format = {
                1: spaces.Box(low=np.zeros(self.obs_dim_map[1]), high=np.ones(self.obs_dim_map[1]), dtype=np.float32),
                2: spaces.Box(low=np.zeros(self.obs_dim_map[2]), high=np.ones(self.obs_dim_map[2]), dtype=np.float32),
                3: spaces.Box(low=np.zeros(self.obs_dim_map[3]), high=np.ones(self.obs_dim_map[3]), dtype=np.float32),
                4: spaces.Box(low=np.zeros(self.obs_dim_map[4]), high=np.ones(self.obs_dim_map[4]), dtype=np.float32)
            }
            self._action_space_in_preferred_format = {1: spaces.MultiDiscrete([13,9,2,2]), 2: spaces.MultiDiscrete([13,9,2]), 3: spaces.MultiDiscrete([13,9,2,2]), 4: spaces.MultiDiscrete([13,9,2])}
        else:
            self.observation_space = spaces.Box(low=np.zeros(self.obs_dim), high=np.ones(self.obs_dim), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([13,9,2,2])

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.obs_dim, dtype=np.float32), 0, {"__all__": False}, {"__all__": False}, {}
    
class EscapeMarl(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True

        self.observation_space = spaces.Box(low=np.zeros(15), high=np.ones(15), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([13,9])

    def reset(self, *, seed=None, options=None):
        return np.zeros(15, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(15, dtype=np.float32), 0, {"__all__": False}, {"__all__": False}, {}

#class DogfightScenario(MultiAgentEnv, TaskSettableEnv):
class DogfightScenario(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True

        self.sim: Optional[CmanoSimulator] = None
        self.map_limits = MapLimits(7.0, 5.0, 7.5, 5.5)

        self.args = env_config.get("args", None)
        self.num_opps = env_config.get("num_opps",3)
        self.num_agents = env_config.get("num_agents",3)
        self.total_num = self.num_agents + self.num_opps
        self.alive_agents = 0
        self.alive_opps = 0

        self.obs_dim = env_config.get("obs_dim", 46)
        self.horizon = env_config.get("horizon",400)
        self.sequential = env_config.get("sequential", False)
        self.n_sub_steps = env_config.get("sub_steps", 20)
        self.ss = env_config.get("ss", 1)
        self.opp_selection = env_config.get("opp_selection", False)
        
        self.steps = 0
        self.opp_stats = {}
        self.opp_to_attack = {}
        self.opp_to_escape = {}
        self.surrounding_dists = {}
        self.closest_opps = {}
        self.missile_wait = {}

        self.observation_space = spaces.Box(low=np.zeros(self.obs_dim), high=np.ones(self.obs_dim), dtype=np.float32)
        if self.opp_selection:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Discrete(2)
        self._agent_ids = set(range(1,self.num_agents+1))

        # Plotting
        self.plt_cfg = PlotConfig()
        self.plt_cfg.units_scale = 20.0
        self.plotter = ScenarioPlotter(self.map_limits, dpi=200, config=self.plt_cfg)

        self.policy_fight = self._get_policy("fight")
        self.policy_escape = self._get_policy("escape")

        self.commander_rewards = {}
        self.keep_steps = 0
        self.keeping = False
        self.min_steps = 10 #min number of low-level steps

    def reset(self, *, seed=None, options=None):

        # self.num_agents = random.randint(2,5)
        # self.num_opps = random.randint(2,5)
        # self.total_num = self.num_agents + self.num_opps
        # self._agent_ids = set(range(1,self.num_agents+1))

        self.keeping = False
        self.keep_steps = 0
        self.min_steps = random.randint(10,15)
        self.steps = 0
        self.alive_agents = 0
        self.alive_opps = 0
        self.missile_wait = {i:0 for i in range(1, self.total_num+1)}
        self.opp_to_attack = {i:None for i in range(1, self.total_num+1)}
        #self.opp_to_escape = {i:None for i in range(1, self.num_agents)}
        #self.opp_stats = {i:None for i in range(1, self.num_agents+1)}
        #self.surrounding_dists = {i:None for i in range(1, self.num_agents+1)}
        self.sim = CmanoSimulator(num_units=self.num_agents, num_opp_units=self.num_opps)
        self._reset_scenario()
        self.commander_rewards = {i:0 for i in range(1,self.num_agents+1)}
        return self._state_hier() if self.ss == 1 else self._state_hier_light(), {}

    def step(self, action):
        """
        Forward current action to sub-env, update sim. 
        """
        if action:
            self._sub_steps(action)

        terminateds = truncateds = {}
        terminateds["__all__"] = self.alive_agents <= 0 or self.alive_opps <= 0 or self.steps >= self.horizon
        truncateds["__all__"] = self.steps >= self.horizon or self.alive_agents <= 0 or self.alive_opps <= 0 
        return self._state_hier() if self.ss == 1 else self._state_hier_light(), self.commander_rewards, terminateds, truncateds, {}

    def _sub_steps(self, commander_action):
        s = 0
        event = False
        self.opp_to_attack = {i:None for i in range(1, self.total_num+1)}
        #self.opp_stats = {i:None for i in range(1, self.num_agents+1)}
        #self.opp_to_escape = {i:None for i in range(1, self.num_agents+1)}
        self.commander_rewards = {i:0 for i in range(1,self.num_agents+1)}
        opp_behaviour = {i: "escape" if self.steps % 20 in range(7) and bool(random.randint(0,1)) else "fight" for i in range(self.num_agents+1, self.total_num+1)}
        self.min_steps = random.randint(10,15)

        self._situation_reward(commander_action)

        while s <= self.n_sub_steps and not event:
            rewards = {i:0 for i in range(1, self.num_agents+1)}
            for i in range(1, self.total_num+1):
                if self.sim.unit_exists(i):
                    u = self.sim.get_unit(i)
                    if i <= self.num_agents:
                        actions = self._sp_actions(i, u, u.ac_type, policy_type="escape" if commander_action[i]==0 else "fight", commander=commander_action[i]-1 if self.opp_selection else None)
                        #actions = self._sp_actions(i, u, u.ac_type, "fight")
                    else:
                        actions = self._sp_actions(i, u, u.ac_type, policy_type=opp_behaviour[i])

                    u.set_heading((u.heading + (actions[i][0]-6)*15)%360)
                    u.set_speed(100+((u.max_speed-100)/8)*actions[i][1])
                    if bool(actions[i][2]) and u.cannon_remain_secs > 0:
                        u.fire_cannon()
                    if u.ac_type == 1:
                        if bool(actions[i][3]) and u.missile_remain > 0 and self.opp_to_attack[i] and not u.actual_missile and self.missile_wait[i] == 0:
                            u.fire_missile(u, self.sim.get_unit(self.opp_to_attack[i]), self.sim)
                            self.missile_wait[i] = random.randint(8, 12)

                    if self.missile_wait[i] > 0 and not bool(u.actual_missile):
                        self.missile_wait[i] = self.missile_wait[i] -1

            rewards, event = self._get_rewards(rewards, self.sim.do_tick())
            if not event:
                #mind.5, mind.10 at critical pos.
                # if self.keeping and abs(self.keep_steps-self.steps)>self.min_steps:
                #     self.keeping = False
                # if not self.keeping and abs(self.keep_steps-self.steps)>5:
                #     event = self._surrounding_event()
                

                if abs(self.keep_steps-self.steps)>self.min_steps:
                    event = self._surrounding_event()
            s += 1
            self.steps += 1
            for k, v in rewards.items():
                self.commander_rewards[k] += v
        return
    
    def _situation_reward(self, commander):
        def eval_attack(ag_id, opp_id, comm=None):
            if self.surrounding_dists[ag_id][comm if comm else 0] < 0.1: #was 0.08
                if self._focus_angle(ag_id, opp_id) < 30:
                    a_a = self._aspect_angle(opp_id, ag_id)
                    if a_a < 0.25 or a_a > 0.75:
                        self.commander_rewards[ag_id] += 1
                        self.min_steps = 35
                    else:
                        self.commander_rewards[ag_id] += 0.5
                        if self.min_steps < 35:
                            self.min_steps = 20 #was 25
                    self.keep_steps = self.steps

        def eval_esc(ag_id):
            num = 0
            for idx, d in enumerate(self.surrounding_dists[ag_id]):
                if d < 0.1 and self._focus_angle(self.closest_opps[ag_id][idx], ag_id) < 30:
                    num += 1
            if num >= 2:
                self.commander_rewards[ag_id] += 0.5
                if self.min_steps < 25:
                    self.min_steps = 15 #was 20
                self.keep_steps = self.steps
            elif self.surrounding_dists[ag_id][0] > 0.2:
                self.commander_rewards[ag_id] -= 0.5

        for i in range(1, self.num_agents+1):
            if self.closest_opps[i]:
                if self.opp_selection:
                    if commander[i] == 0:
                        try:
                            eval_esc(i)
                        except:
                            pass
                    else:
                        try:
                            eval_attack(i, self.closest_opps[i][commander[i]-1], comm=commander[i]-1)
                        except:
                            pass

                else:
                    if eval_attack(i, self.closest_opps[i][0]) and commander[i]==1:
                        self.commander_rewards[i] += 0.4

    def _surrounding_event(self):
        def eval_event(ag_id, opp_id, idx):
            if self.surrounding_dists[ag_id][idx] < 0.1: #was 0.08
                if self._focus_angle(ag_id, opp_id) < 30 or self._focus_angle(opp_id, ag_id) < 30:
                    return True
            return False
        
        event = False
        for i in range(1, self.num_agents+1):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                if x < 0.15 or x > 0.85 or y < 0.15 or y > 0.85:
                    event = True
                if not event:
                    if self.closest_opps[i]:
                        for o_idx, o in enumerate(self.closest_opps[i]):
                            if not self.opp_selection:
                                if o_idx <= 0:
                                    event =  eval_event(i, o, o_idx)
                            else:
                                event =  eval_event(i, o, o_idx)
                if not event:
                    k = 0
                    if self.surrounding_dists[i]:
                        for j, d in enumerate(self.surrounding_dists[i]):
                            if d < 0.1 and self._focus_angle(self.closest_opps[i][j], i) < 30:
                                k+= 1
                    if k >= 2:
                        event = True
        if event:
            self.keep_steps = self.steps
        return event

    def _get_rewards(self, rewards, events):
        event = False
        for ev in events:
            if ev.unit_killer.id <= self.num_agents:
                if ev.unit_destroyed.id in range(self.num_agents+1, self.total_num+1):

                    rewards[ev.unit_killer.id] += 2
                    self.alive_opps -= 1

            # killed by opp
            elif ev.unit_destroyed.id <= self.num_agents and ev.unit_killer.id in range(self.num_agents+1, self.total_num+1):
                rewards[ev.unit_destroyed.id] -= 2
                self.alive_agents -= 1

            event = True

        for i in range(1, self.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if not self.map_limits.in_boundary(u.position.lat, u.position.lon):
                    self.sim.remove_unit(i)
                    if i <= self.num_agents:
                        rewards[i] -= 2
                        self.alive_agents -= 1
                    else:
                        self.alive_opps -= 1
                    event = True

        return rewards, event

    def _get_policy(self, policy_type):
        if policy_type=="fight":
            ModelCatalog.register_custom_model(f"ac1_model",CCAtt1)
            ModelCatalog.register_custom_model(f"ac2_model",CCAtt2)
            #check_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'HETERO', 'SS2_HETERO_Att', 'SS2_L4_HETERO_Att', 'checkpoint')
            check_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'hier_fight', 'checkpoint')
            return Policy.from_checkpoint(check_path, ["ac1_policy", "ac2_policy"])
        else:
            ModelCatalog.register_custom_model(f"ac1_model",CCEsc1)
            ModelCatalog.register_custom_model(f"ac2_model",CCEsc2)
            check_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'HETERO', 'SS2_HETERO_ESC', 'SS2_L3_HETERO_ESC', 'checkpoint')
            return Policy.from_checkpoint(check_path, ["ac1_policy", "ac2_policy"])

    def _sp_actions(self, agent_id, unit, ac_type, policy_type="fight", commander=None):
        def cc_obs(obs, esc=False):
            if ac_type == 1:
                return {
                    "obs_1_own": obs,
                    "obs_2": np.zeros(28+int(esc), dtype=np.float32),
                    "obs_3": np.zeros(30, dtype=np.float32),
                    "obs_4": np.zeros(28+int(esc), dtype=np.float32),
                    "act_1_own": np.zeros(ACTION_DIM_AC1),
                    "act_2": np.zeros(ACTION_DIM_AC2),
                    "act_3": np.zeros(ACTION_DIM_AC1),
                    "act_4": np.zeros(ACTION_DIM_AC2),
                }
            else:
                return {
                    "obs_1_own": obs,
                    "obs_2": np.zeros(30, dtype=np.float32),
                    "obs_3": np.zeros(30, dtype=np.float32),
                    "obs_4": np.zeros(28+int(esc), dtype=np.float32),
                    "act_1_own": np.zeros(ACTION_DIM_AC2),
                    "act_2": np.zeros(ACTION_DIM_AC1),
                    "act_3": np.zeros(ACTION_DIM_AC1),
                    "act_4": np.zeros(ACTION_DIM_AC2),
                }
        actions = {}

        if policy_type == "escape":
            state = self._esc_state(agent_id, unit)
            a = self.policy_escape[f'ac{ac_type}_policy'].compute_single_action(obs=cc_obs(state, True), state=torch.zeros(1))
            actions[agent_id] = a[0]

        else:
            if commander:
                try:
                    opp_id = self.closest_opps[agent_id][commander]
                except:
                    opp_id = None
                state = self._fight_state(agent_id, unit, opp_id)
            else:
                state = self._fight_state(agent_id, unit)
            a = self.policy_fight[f'ac{ac_type}_policy'].compute_single_action(obs=cc_obs(state), state=torch.zeros(1))
            actions[agent_id] = a[0]

        return actions

    def _state_hier_light(self):
        def hier_opp_state(opp_id, agent_id, dist):
            state = []
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(self._focus_angle(agent_id, opp_id, True))
            state.append(self._focus_angle(opp_id, agent_id, True))
            state.append(self._aspect_angle(opp_id, agent_id))
            state.append(self._aspect_angle(agent_id, opp_id))
            state.append(dist)
            return state
        
        def hier_fri_state(agent_id, oth_agent, dist):
            unit = self.sim.get_unit(oth_agent)
            state = []
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(dist)
            return state

        state_dict = {}
        self.surrounding_dists = {i:None for i in range(1, self.num_agents+1)}
        self.closest_opps = {i:None for i in range(1, self.num_agents+1)}

        for ag_id in range(1, self.num_agents+1):
            state = []
            if self.sim.unit_exists(ag_id):
                d_opps = self._closest_object(ag_id)
                if d_opps:
                    unit = self.sim.get_unit(ag_id)
                    x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                    state.append(x)
                    state.append(y)
                    state.append(np.clip(unit.speed/unit.max_speed,0,1))
                    state.append(np.clip((unit.heading%359)/359, 0, 1))
                    opp_state = []
                    for d in d_opps:
                        opp_s = hier_opp_state(d[0], ag_id, d[1])
                        opp_state.extend(opp_s)
                        if len(opp_state) == 27:
                            break
                    if len(opp_state) < 27:
                        opp_state.extend(np.zeros(27-len(opp_state)))
                    fri_state = []
                    d_fri = self._closest_object(ag_id, True)
                    for f in d_fri:
                        fri_s = hier_fri_state(ag_id, f[0], f[1])
                        fri_state.extend(fri_s)
                        if len(fri_state) == 10:
                            break
                    if len(fri_state) < 10:
                        fri_state.extend(np.zeros(10-len(fri_state)))
                    state.extend(opp_state)
                    state.extend(fri_state)
                    assert len(state) == self.obs_dim
                    dists = []
                    cl_opps = []
                    for d in d_opps:
                        dists.append(d[2])
                        cl_opps.append(d[0])
                    self.surrounding_dists[ag_id] = dists
                    self.closest_opps[ag_id] = cl_opps
                else:
                    state = np.zeros(self.obs_dim, dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim, dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _state_hier(self,):
        """
        Current observation, stored in state_dict, only containing alive agents.
        """
        def hier_opp_state(opp_id, agent_id, dist):
            state = []
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(self._focus_angle(agent_id, opp_id, True))
            state.append(self._focus_angle(opp_id, agent_id, True))
            state.append(self._aspect_angle(opp_id, agent_id))
            state.append(dist)
            return state
        
        def hier_fri_state(agent_id, oth_agent, dist):
            unit = self.sim.get_unit(oth_agent)
            state = []
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(self._heading_diff(agent_id, oth_agent))
            state.append(self._focus_angle(agent_id, oth_agent, True))
            state.append(self._focus_angle(oth_agent, agent_id, True))
            state.append(dist)
            return state

        state_dict = {}
        self.surrounding_dists = {i:None for i in range(1, self.num_agents+1)}

        for ag_id in range(1, self.num_agents+1):
            state = []
            if self.sim.unit_exists(ag_id):
                d_opps = self._closest_object(ag_id)
                if d_opps:
                    unit = self.sim.get_unit(ag_id)
                    x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                    state.append(x)
                    state.append(y)
                    state.append(np.clip(unit.speed/unit.max_speed,0,1))
                    state.append(np.clip((unit.heading%359)/359, 0, 1))
                    state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
                    if unit.ac_type == 1:
                        state.append(np.clip(unit.missile_remain/unit.rocket_max, 0, 1))
                        state.append(int(bool(unit.actual_missile or unit.cannon_current_burst_secs > 0)))
                    else:
                        state.append(0)
                        state.append(int(unit.cannon_current_burst_secs > 0))
                    opp_state = []
                    for d in d_opps:
                        opp_s = hier_opp_state(d[0], ag_id, d[1])
                        opp_state.extend(opp_s)
                        if len(opp_state) == 24:
                            break
                    if len(opp_state) < 24:
                        opp_state.extend(np.zeros(24-len(opp_state)))
                    fri_state = []
                    d_fri = self._closest_object(ag_id, True)
                    for f in d_fri:
                        fri_s = hier_fri_state(ag_id, f[0], f[1])
                        fri_state.extend(fri_s)
                        if len(fri_state) == 16:
                            break
                    if len(fri_state) < 16:
                        fri_state.extend(np.zeros(16-len(fri_state)))
                    state.extend(opp_state)
                    state.extend(fri_state)
                    assert len(state) == self.obs_dim
                    dists = []
                    for d in d_opps:
                        dists.append(d[2])
                    self.surrounding_dists[ag_id] = dists
                else:
                    state = np.zeros(self.obs_dim, dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim, dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _fight_state(self, ag_id, unit, opp_id=None):
        """
        Current observation, stored in state_dict, only containing alive agents.
        """

        def _opp_state(opp_id, agent_id, dist):
            """
            Depending on state-space option, compute the opponent part of the whole observation. 
            """
            state = []
            unit = self.sim.get_unit(opp_id)
            x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(self._focus_angle(opp_id, agent_id,True))
            state.append(self._aspect_angle(agent_id, opp_id))
            state.append(self._heading_diff(opp_id, agent_id))
            state.append(dist)
            state.append(int(unit.cannon_current_burst_secs > 0))
            if unit.ac_type==1:
                state.append(int(bool(unit.actual_missile)))
            else:
                state.append(0)
            return state

        def _oth_agent_state(act_agent_id, oth_agent):
            unit = self.sim.get_unit(oth_agent)
            state = []
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(self._heading_diff(act_agent_id, oth_agent))
            state.append(self._focus_angle(act_agent_id, oth_agent, True))
            state.append(self._focus_angle(oth_agent, act_agent_id, True))
            state.append(self._distance(act_agent_id, oth_agent, True))
            state.append(int(bool(unit.cannon_current_burst_secs > 0) or bool(unit.actual_missile)))
            return state

        state = []
        self.opp_to_attack[ag_id] = None
        d_opps = []
        if opp_id and self.sim.unit_exists(opp_id):
            d_opps.append(opp_id)
            d_opps.append(self._distance(ag_id, opp_id, True))
        else:
            d_closest = self._closest_object(ag_id)
            if d_closest:
                d_opps.append(d_closest[0][0])
                d_opps.append(d_closest[0][1])

        if d_opps:
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(self._focus_angle(ag_id, d_opps[0],True))
            state.append(self._aspect_angle(d_opps[0], ag_id))
            state.append(self._heading_diff(ag_id, d_opps[0]))
            state.append(d_opps[1])
            state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
            if unit.ac_type == 1:
                state.append(np.clip(unit.missile_remain/(unit.rocket_max+1e-9), 0, 1))
                state.append(int(self.missile_wait[ag_id]==0))
                state.append(int(bool(unit.actual_missile)))
            else:
                state.append(int(unit.cannon_current_burst_secs > 0))
            state.extend(_opp_state(d_opps[0], ag_id, d_opps[1]))
            d_fri = self._closest_object(ag_id, True)
            if d_fri:
                state.extend(_oth_agent_state(ag_id, d_fri[0][0]))
            else:
                state.extend(np.zeros(8))
            self.opp_to_attack[ag_id] = d_opps[0]
        else:
            state = np.zeros(30 if unit.ac_type==1 else 28, dtype=np.float32)

        state = np.array(state, dtype=np.float32)
        return state

    def _esc_state(self, ag_id, unit):
        """
        Observation for escaping.
        """
        def esc_opp_state(opp_id, agent_id, dist):
            state = []
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(self._heading_diff(agent_id, opp_id))
            state.append(self._focus_angle(agent_id, opp_id, True))
            state.append(self._focus_angle(opp_id, agent_id, True))
            state.append(dist)
            return state
        
        def esc_fri_state(agent_id, oth_agent):
            unit = self.sim.get_unit(oth_agent)
            state = []
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(self._heading_diff(agent_id, oth_agent))
            state.append(self._focus_angle(agent_id, oth_agent, True))
            state.append(self._focus_angle(oth_agent, agent_id, True))
            state.append(self._distance(agent_id, oth_agent, True))
            return state
            
        state = []
        #self.opp_to_escape[ag_id] = None
        d_opps = self._closest_object(ag_id)
        if d_opps:
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(np.clip(unit.speed/unit.max_speed,0,1))
            state.append(np.clip((unit.heading%359)/359, 0, 1))
            state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
            if unit.ac_type==1:
                state.append(np.clip(unit.missile_remain/(unit.rocket_max+1e-9), 0, 1))
            opp_state = []
            for d in d_opps:
                opp_s = esc_opp_state(d[0], ag_id, d[1])
                opp_state.extend(opp_s)
                if len(opp_state) == 16:
                    break
            if len(opp_state) < 16:
                opp_state.extend(np.zeros(16-len(opp_state)))
            state.extend(opp_state)
            d_fri = self._closest_object(ag_id, True)
            if d_fri:
                state.extend(esc_fri_state(ag_id, d_fri[0][0]))
            else:
                state.extend(np.zeros(8))
            # try:
            #     self.opp_to_escape[ag_id] = d_opps[:2]
            # except:
            #     self.opp_to_escape[ag_id] = d_opps
        else:
            state = np.zeros(30 if unit.ac_type==1 else 29, dtype=np.float32)

        state = np.array(state, dtype=np.float32)

        return state

    def _closest_object(self, agent_id, friendly=False):
        """
        Return a sorted list with id's and distances to opponents. 
        """
        order = []
        if friendly:
            f = list(range(1,self.num_agents+1)) if agent_id <= self.num_agents else list(range(self.num_agents+1, self.total_num+1))
            f.remove(agent_id)
            for i in f:
                if self.sim.unit_exists(i):
                    order.append([i, self._distance(agent_id, i, True)])
        else:
            if agent_id <= self.num_agents:
                start = self.num_agents + 1
                end = self.total_num + 1
            else:
                start = 1
                end = self.num_agents + 1
            for i in range(start, end):
                if self.sim.unit_exists(i):
                    order.append([i, self._distance(agent_id, i, True), self._distance(agent_id, i)])
        order.sort(key=lambda x:x[1])
        return order

    def _focus_angle(self, agent_id, opp_id, norm=False):
        """
        Compute focus angle based on vector angles of current heading direction and position of the two airplanes. 
        """
        x = np.clip((np.dot(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ),sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]), np.array([self.sim.get_unit(opp_id).position.lon-self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat-self.sim.get_unit(agent_id).position.lat])))/(np.linalg.norm(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ),sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]))*np.linalg.norm(np.array([self.sim.get_unit(opp_id).position.lon-self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat-self.sim.get_unit(agent_id).position.lat]))+1e-10), -1, 1)
        if norm:
            return np.clip( (acos(x) * (180 / pi))/180, 0, 1)
        else:
            return acos(x) * (180 / pi)

    def _distance(self, agent_id, opp_id, norm=False):
        """
        Euclidian Distance between two aircrafts.
        """
        d = hypot(self.sim.get_unit(opp_id).position.lon - self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat - self.sim.get_unit(agent_id).position.lat)
        return self._shifted_range(d, 0, sqrt(0.5), 0, 1) if norm else d
        
    def _correct_angle_sign(self, opp_unit, ag_unit):
        """
        The correct heading is computed in 'hardcoded_opps'. 
        Here the correct direction is determined, if to turn right or left. 
        """
        def line(x0, y0, x1, y1, x2, y2):
            return  (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)

        x = opp_unit.position.lon
        y = opp_unit.position.lat
        a = opp_unit.heading
        x1 = x + round(sin(radians(a%360)),3)
        y1 = y + round(cos(radians(a%360)),3)

        if ag_unit:
            xc = ag_unit.position.lon
            yc = ag_unit.position.lat
            val = line(x,y,x1,y1,xc,yc)
            if val < 0:
                return 1
            else:
                return -1
        else:
            return 1

    def _escaping_opp(self, unit):
        """
        This ensures that the hardcoded opponents don't stuck in rotating around agents. 
        So, they shortly escape in the diagonal direction.
        """
        y, x = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        if y < 0.5:
            if x < 0.5:
                heading = int(random.uniform(30, 60))
            else:
                heading = int(random.uniform(300, 330))
        else:
            if x < 0.5:
                heading = int(random.uniform(120, 150))
            else:
                heading = int(random.uniform(210, 240))
        speed= int(random.uniform(300, 600))
        sign = np.sign(heading-unit.heading+1e-9)
        return None, heading, speed, bool(random.randint(0,1)), False, np.clip(((heading-unit.heading)%360)*sign, -180, 180)

    def _sp_opp_escape(self, unit):
        y, x = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        heading = unit.heading
        if self.steps % 30 == 0 or y < 0.15 or y> 0.85 or x < 0.15 or x > 0.85:
            if y < 0.5:
                if x < 0.5:
                    heading = int(random.uniform(30, 60))
                else:
                    heading = int(random.uniform(300, 330))
            else:
                if x < 0.5:
                    heading = int(random.uniform(120, 150))
                else:
                    heading = int(random.uniform(200, 250))
        if y < 0.2 or y> 0.8 or x < 0.2 or x > 0.8:
            speed = int(random.uniform(100, 150))
        else:
            speed = int(random.uniform(300, 800)) if unit.ac_type==1 else int(random.uniform(300, 600))
        sign = np.sign(heading-unit.heading+1e-9)
        heading_rel = np.clip(((heading-unit.heading)%360)*sign, -180, 180)
        return self._shifted_range(heading_rel, -180,180, -90,90), speed, bool(random.randint(0,1)), bool(random.randint(0,1)) if unit.ac_type==1 else False

    def _hardcoded_opp(self, opp_id):
        """
        Deterministic opponents to fight against agents. They have slight randomness in heading and speed. 
        """
        d_agt = self._closest_object(opp_id)
        opp_unit = self.sim.get_unit(opp_id)
        heading = opp_unit.heading
        fire = False
        fire_missile = False
        speed = int(random.uniform(100, 400))
        head_rel = 0 #heading degree towards opp
        if d_agt:
            sign = self._correct_angle_sign(opp_unit, self.sim.get_unit(d_agt[0][0]))
            r = random.uniform(0.7, 1.3)
            focus = self._focus_angle(opp_id, d_agt[0][0])
            if d_agt[0][1] > 0.008 and focus > 4:
                head_rel = r*sign*focus
                heading = (heading + r*sign*focus)%360
            if d_agt[0][1] > 0.05:
                speed = int(random.uniform(500, 800)) if focus < 30 else int(random.uniform(100, 500))
            fire = d_agt[0][1] < 0.03 and focus < 10
            fire_missile = d_agt[0][1] < 0.09 and focus < 5 # ROCKET
            speed = np.clip(speed, 0, 600) if opp_unit.ac_type == 2 else speed
            return d_agt[0][0], heading, speed, fire, fire_missile, head_rel
        speed = np.clip(speed, 0, 600) if opp_unit.ac_type == 2 else speed
        return None, heading, speed, fire, fire_missile, head_rel

    def _aspect_angle(self, agent_id, opp_id, norm=True):
        """
        Aspect angle: angle from agent_id tail to opp_id, regardless of heading of opp_id.
        """
        focus = self._focus_angle(agent_id, opp_id)
        sign = self._correct_angle_sign(self.sim.get_unit(agent_id), self.sim.get_unit(opp_id))
        return (abs((focus*sign)-180)%359)/359 if norm else abs((focus*sign)-180)%359

    def _heading_diff(self, agent_id, opp_id, norm=True):
        """
        Angle between heading vectors.
        """
        x = np.clip((np.dot(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ), sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]), np.array([cos( ((90-self.sim.get_unit(opp_id).heading)%360)*(pi/180) ), sin( ((90-self.sim.get_unit(opp_id).heading)%360)*(pi/180) )])))/(np.linalg.norm(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ),sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]))*np.linalg.norm(np.array([cos( ((90-self.sim.get_unit(opp_id).heading)%360)*(pi/180) ), sin( ((90-self.sim.get_unit(opp_id).heading)%360)*(pi/180) )]))+1e-10), -1, 1)
        if norm:
            return np.clip( (acos(x) * (180 / pi))/180, 0, 1)
        else:
            return acos(x) * (180 / pi)

    def _shifted_range(self, x, a,b, c,d):
        """
        find value in new range from [a,b] to [c,d]
        """
        return c + ((d-c)/(b-a))*(x-a)

    def _sample_state(self, agent, i, r):
        x = 0
        y = 0
        a = 0
        if agent == "agent":
            if r == 1:
                x = random.uniform(7.07, 7.22)
                y = random.uniform(5.07 + i*(0.4/self.num_agents), 5.12 + i*(0.4/self.num_agents))
                a = random.randint(0, 359)
            elif r == 2:
                x = random.uniform(7.28, 7.43)
                y = random.uniform(5.07 + i*(0.4/self.num_agents), 5.12 + i*(0.4/self.num_agents))
                a = random.randint(0, 359)

        else:
            if r == 1:
                x = random.uniform(7.28, 7.43)
                y = random.uniform(5.07 + i*(0.4/self.num_opps), 5.12 + i*(0.4/self.num_opps))
                a = random.randint(0, 359)
            elif r == 2:
                x = random.uniform(7.07, 7.22)
                y = random.uniform(5.07 + i*(0.4/self.num_opps), 5.12 + i*(0.4/self.num_opps))
                a = random.randint(0, 359)
        
        return x,y,a

    def _reset_scenario(self):
        """
        create instances of airplane units (Rafale) and store them in dicts self.agents / self.opps.
        when instances get shot, they won't be removed / deleted, but just get invisible.
        """
        r = random.randint(1,2) #chose sides
        for i in range(self.num_agents):
            x, y, a = self._sample_state("agent", i, r)

            ac = random.randint(1,2)
            if ac == 1:
                agent_unit = Rafale(Position(y, x, 10_000), heading=a, speed=100, group="agent", friendly_check=True)
                agent_unit.missile_remain = 8
                agent_unit.rocket_max = 8
                agent_unit.cannon_range_km = 3.0
            else:
                agent_unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=100, group="agent", friendly_check=True)

            agent_unit.cannon_max = 300
            agent_unit.cannon_remain_secs = 300
            self.sim.add_unit(agent_unit)
            self.sim.record_unit_trace(agent_unit.id)
            self.alive_agents += 1

        for i in range(self.num_opps):
            x, y, a = self._sample_state("opp", i, r)

            ac = random.randint(1,2)
            if ac == 1:
                opp_unit = Rafale(Position(y, x, 10_000), heading=a, speed=100, group="opp", friendly_check=True)
                opp_unit.missile_remain = 8
                opp_unit.rocket_max = 8
            else:
                opp_unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=100, group="opp", friendly_check=True)
                
            opp_unit.cannon_max = 300
            opp_unit.cannon_remain_secs = 300
            self.sim.add_unit(opp_unit)
            self.sim.record_unit_trace(opp_unit.id)
            self.alive_opps += 1

    def plot(self, out_file: Path, paths=True):
        objects = [
            StatusMessage(self.sim.status_text),
            TopLeftMessage(self.sim.utc_time.strftime("%Y %b %d %H:%M:%S"))
        ]
        for i in range(1, self.num_agents+self.num_opps+1):
            col = 'blue' if i<=self.num_agents else 'red'
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                objects.extend(self._plot_airplane(unit, col, paths))
            else:
                objects.extend(self._plot_airplane(None, col, paths, True, i))
        for i in range(self.total_num+1, self.total_num*5+2):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                col = "blue" if unit.source.id <= self.num_agents else "red"
                objects.append(
                    Missile(unit.position.lat, unit.position.lon, unit.heading, edge_color=colors[f'{col}_outline'], fill_color=colors[f'{col}_fill'],
                    info_text=f"m_{i}", zorder=0),
                )
        self.plotter.to_png(str(out_file), objects)

    def _plot_airplane(self, a: Rafale, side: str, path=True, use_backup=False, u_id=0) -> List[Drawable]:
        objects = []
        if use_backup:
            trace = [(position.lat, position.lon) for t, position, heading, speed in self.sim.trace_record_units[u_id]]
            objects.append(PolyLine(trace, line_width=1, dash=(2, 2),
                                    edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                    zorder=0))
            objects.append(Waypoint(trace[-1][0], trace[-1][1],
                                edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                fill_color=colors['red_fill'] if side == 'red' else colors['blue_fill'],
                                info_text=f"r_{u_id}", zorder=0))
        else:
            objects = [Airplane(a.position.lat, a.position.lon, a.heading,
                                edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                fill_color=colors['red_fill'] if side == 'red' else colors['blue_fill'],
                                info_text=f"r_{a.id}", zorder=0)]
            if path:
                trace = [(position.lat, position.lon) for t, position, heading, speed in self.sim.trace_record_units[a.id]]
                objects.append(PolyLine(trace, line_width=1, dash=(2, 2),
                                        edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                        zorder=0))
            if a.cannon_current_burst_secs > 0:  # noqa
                d1 = geodetic_direct(a.position.lat, a.position.lon,
                                    sum_angles(a.heading, a.cannon_width_deg / 2.0),
                                    a.cannon_range_km * 1000)
                d2 = geodetic_direct(a.position.lat, a.position.lon,
                                    sum_angles(a.heading, - a.cannon_width_deg / 2.0),
                                    a.cannon_range_km * 1000)
                objects.append(PolyLine([(a.position.lat, a.position.lon),
                                        (d1[0], d1[1]), (d2[0], d2[1]),
                                        (a.position.lat, a.position.lon)], line_width=1, dash=(1, 1),
                                        edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                        zorder=0))
        return objects
