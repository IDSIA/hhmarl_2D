"""
Low Level Environment for MARL Warsim, where 2 groups of airplanes fight against each other. 
"""

import torch
import os
import random
import numpy as np
from math import sin, cos, acos, pi, hypot, radians, exp, sqrt
from gymnasium import spaces
from pathlib import Path
from typing import List, Dict, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from warsim.scenplotter.scenario_plotter import PlotConfig, ColorRGBA, StatusMessage, TopLeftMessage, \
    Airplane, PolyLine, Drawable, Waypoint, Missile, ScenarioPlotter

from warsim.simulator.cmano_simulator import Position, CmanoSimulator, UnitDestroyedEvent
from warsim.simulator.ac1 import Rafale
from warsim.simulator.ac2 import RafaleLong
from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits
from utils.angles import sum_angles

from models.ac_models_hetero import Fight1, Fight2, Esc1, Esc2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

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

class DogfightScenario(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True

        self.sim: Optional[CmanoSimulator] = None
        self.map_limits = MapLimits(7.0, 5.0, 7.3, 5.3)

        self.args = env_config.get("args", None)
        self.num_opps = env_config.get("num_opps",2)
        self.num_agents = env_config.get("num_agents",2)
        self.total_num = self.num_agents + self.num_opps
        self.alive_agents = 0
        self.alive_opps = 0

        self.curr_level = env_config.get("level",1)
        self.agent_mode = env_config.get("agent_mode", "fight")
        self.opp_mode = "fight"
                
        self.steps = 0
        self.opp_to_attack = {}
        self.opp_to_escape = {}
        self.missile_wait = {}
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.horizon_level = {1: 150, 2:200, 3:300, 4:350, 5:400}
        self.obs_dim_map = {1: 30, 2:28, 3:30, 4:28} if self.agent_mode == "fight" else {1: 30, 2:29, 3:30, 4:29}

        self._obs_space_in_preferred_format = {
            1: spaces.Box(low=np.zeros(self.obs_dim_map[1]), high=np.ones(self.obs_dim_map[1]), dtype=np.float32),
            2: spaces.Box(low=np.zeros(self.obs_dim_map[2]), high=np.ones(self.obs_dim_map[2]), dtype=np.float32),
            3: spaces.Box(low=np.zeros(self.obs_dim_map[3]), high=np.ones(self.obs_dim_map[3]), dtype=np.float32),
            4: spaces.Box(low=np.zeros(self.obs_dim_map[4]), high=np.ones(self.obs_dim_map[4]), dtype=np.float32)
        }
        self._action_space_in_preferred_format = {1: spaces.MultiDiscrete([13,9,2,2]), 2: spaces.MultiDiscrete([13,9,2]), 3: spaces.MultiDiscrete([13,9,2,2]), 4: spaces.MultiDiscrete([13,9,2])}
        self._agent_ids = set(range(1,self.total_num+1))

        # Plotting
        self.plt_cfg = PlotConfig()
        self.plt_cfg.units_scale = 20.0
        self.plotter = ScenarioPlotter(self.map_limits, dpi=200, config=self.plt_cfg)

        self.sp_opp = None
        if self.curr_level == 4:
           self.sp_opp = self._get_policy()

        elif self.curr_level == 5:
            self.sp_opps = {}
            self._setup_sp_history()

    def reset(self, *, seed=None, options=None):
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.steps = 0
        self.alive_agents = 0
        self.alive_opps = 0
        self.missile_wait = {i:0 for i in range(1, self.total_num+1)}
        self.opp_to_attack = {i:None for i in range(1, self.total_num+1)}
        self.opp_to_escape = {i:None for i in range(1, self.total_num+1)}
        self.sim = CmanoSimulator(num_units=self.num_agents, num_opp_units=self.num_opps)
        self._reset_scenario()
        self.horizon = self.horizon_level[self.curr_level]
        opp_actions = {3:[0,0,0,0], 4:[0,0,0]}
        if self.curr_level == 5:
            k = random.randint(1,5)
            self.sp_opp = self.sp_opps[k]
            self.opp_mode = "escape" if k == 5 else "fight"
        return self._fight_state() if self.agent_mode == "fight" else self._esc_state(), {1:opp_actions, 2:opp_actions}

    def step(self, action):
        """
        Take a step for all agents in env.
        """
        self.steps += 1
        reward = {}
        opp_actions = {}

        if action:
            reward, opp_actions = self._take_action(action)

        terminateds = truncateds = {}
        truncateds["__all__"] = terminateds["__all__"] = self.alive_agents <= 0 or self.alive_opps <= 0 or self.steps >= self.horizon
        return self._fight_state() if self.agent_mode == "fight" else self._esc_state(), reward, terminateds, truncateds, {1:opp_actions, 2:opp_actions}

    def _take_action(self, action):
        """
        Apply actions to agents and opponents and get rewards.
        Opponent behavior is scripted: L1 and L2 random, L3 engage closest agent, L4 previous policy, L5 all previous policies randomly assigned.
        """
        rewards = {}
        opp_stats = {}
        opp_actions = {3:[0,0,0,0], 4:[0,0,0]}

        for i in range(1, self.total_num+1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if i <= self.num_agents or self.curr_level >= 4:
                    if i >= self.num_agents+1:
                        actions = self._sp_actions(i, u)
                        opp_actions[i][0] = actions[i][0]
                        opp_actions[i][1] = actions[i][1]
                        opp_actions[i][2] = actions[i][2]
                        if u.ac_type==1:
                            opp_actions[i][3] = actions[i][3]
                    else:
                        actions = action
                        rewards[i] = 0
                        if self.sim.unit_exists(self.opp_to_attack[i]):
                            opp_stats[i] = [self._focus_angle(self.opp_to_attack[i], i, True), self._distance(i, self.opp_to_attack[i])]

                    u.set_heading((u.heading + (actions[i][0]-6)*15)%360)
                    u.set_speed(100+((u.max_speed-100)/8)*actions[i][1])
                    if bool(actions[i][2]) and u.cannon_remain_secs > 0:
                        u.fire_cannon()
                        if self.agent_mode == "escape" and i <= self.num_agents and u.cannon_remain_secs < 50:
                            rewards[i] = -0.1
                    if u.ac_type == 1:
                        if bool(actions[i][3]) and u.missile_remain > 0 and self.opp_to_attack[i] and not u.actual_missile and self.missile_wait[i] == 0:
                            u.fire_missile(u, self.sim.get_unit(self.opp_to_attack[i]), self.sim)
                            if i <= self.num_agents:
                                self.missile_wait[i] = random.randint(10,25)
                                if self.agent_mode == "escape" and u.missile_remain < 2:
                                    rewards[i] = -0.1
                            else:
                                self.missile_wait[i] = random.randint(5,15)

                else:
                    if self.curr_level == 1 and not u.actual_missile and self.steps % 40 in range(3) and bool(random.randint(0,1)) and self.missile_wait[i] == 0 and u.ac_type == 1:
                        d_ag = self._closest_object(i)
                        if d_ag:
                            u.fire_missile(u, self.sim.get_unit(d_ag[0][0]), self.sim)
                            self.missile_wait[i] = 5
                            opp_actions[i][3] = 1

                    elif self.curr_level == 2:
                        u.fire_cannon()
                        opp_actions[i][2] = 1
                        if self.steps==1 or self.steps % random.randint(35,45) <= 5:
                            r = random.randint(0,1)
                            h = 12 if r==0 else 0
                            u.set_heading((u.heading + ((-1)**r)*90)%360) #12
                            s = 100 + random.randint(0,4)*75
                            u.set_speed(s) # int(s/100)
                            opp_actions[i][0] = h
                            opp_actions[i][1] = int(s/100)
                        if not u.actual_missile and self.steps % 40 in range(3) and bool(random.randint(0,1)) and self.missile_wait[i] == 0 and u.ac_type == 1:
                            d_ag = self._closest_object(i)
                            if d_ag:
                                u.fire_missile(u, self.sim.get_unit(d_ag[0][0]), self.sim)
                                self.missile_wait[i] = 5
                                opp_actions[i][3] = 1

                    elif self.curr_level == 3:
                        if self.steps % 60 == 0 and not self.hardcoded_opps_escaping:
                            self.hardcoded_opps_escaping = bool(random.randint(0, 1))
                            if self.hardcoded_opps_escaping:
                                self.opps_escaping_time = int(random.uniform(20, 30))

                        if self.hardcoded_opps_escaping:
                            opp, heading, speed, fire, fire_missile, head_rel = self._escaping_opp(u)
                            self.opps_escaping_time -= 1
                            if self.opps_escaping_time <= 0:
                                self.hardcoded_opps_escaping = False
                        else:
                            opp, heading, speed, fire, fire_missile, head_rel = self._hardcoded_opp(i)
                        u.set_heading(heading)
                        u.set_speed(speed)
                        head_rel = self._shifted_range(head_rel, -180,180, -90,90)
                        opp_actions[i][0] = int(head_rel/15)+6
                        opp_actions[i][1] = int(speed/100)
                        if fire:
                            u.fire_cannon()
                            opp_actions[i][2] = 1
                        if fire_missile and opp and not u.actual_missile and self.missile_wait[i] == 0 and u.ac_type == 1:
                            u.fire_missile(u, self.sim.get_unit(opp), self.sim)
                            self.missile_wait[i] = 10
                            opp_actions[i][3] = 1

                if self.missile_wait[i] > 0 and not bool(u.actual_missile):
                    self.missile_wait[i] = self.missile_wait[i] -1

        return self._get_rewards(rewards, self.sim.do_tick(), opp_stats), opp_actions

    def _get_rewards(self, rewards, events, opp_stats):
        """
        Calculating Rewards. 
        First check for out-of-boundary, then killing rewards.
        """

        for i in range(1, self.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if not self.map_limits.in_boundary(u.position.lat, u.position.lon):
                    self.sim.remove_unit(i)
                    if i <= self.num_agents:
                        rewards[i] = -5
                        self.alive_agents -= 1
                    else:
                        self.alive_opps -= 1

        if self.agent_mode == "fight":
            for ev in events:

                # agent kill
                if ev.unit_killer.id <= self.num_agents:
                    if ev.unit_destroyed.id in range(self.num_agents+1, self.total_num+1):
                        
                        if ev.origin.id >= self.total_num+1: #killed by rocket
                            rewards[ev.unit_killer.id] = self._shifted_range(ev.unit_killer.missile_remain/ev.unit_killer.rocket_max, 0,1, 1,1.5)
                        else:
                            rewards[ev.unit_killer.id] = self._shifted_range(ev.unit_killer.cannon_remain_secs/ev.unit_killer.cannon_max, 0,1, 0.5,1) + self._shifted_range(opp_stats[ev.unit_killer.id][0], 0,1, 0.5,1) #range [1,2]
                        self.alive_opps -= 1

                    # killed by friend
                    elif ev.unit_destroyed.id <= self.num_agents:
                        rewards[ev.unit_killer.id] = -2
                        self.alive_agents -= 1

                # killed by opp
                elif ev.unit_destroyed.id <= self.num_agents and ev.unit_killer.id in range(self.num_agents+1, self.total_num+1):
                    rewards[ev.unit_destroyed.id] = -2
                    self.alive_agents -= 1
        else:
            for ev in events:

                # agent kill
                if ev.unit_killer.id <= self.num_agents:
                    if ev.unit_destroyed.id in range(self.num_agents+1, self.total_num+1):
                        rewards[ev.unit_killer.id] = 1
                        self.alive_opps -= 1
                    elif ev.unit_destroyed.id <= self.num_agents:
                        rewards[ev.unit_killer.id] = -2
                        self.alive_agents -= 1

                # killed by opp
                elif ev.unit_destroyed.id <= self.num_agents and ev.unit_killer.id in range(self.num_agents+1, self.total_num+1):
                    rewards[ev.unit_destroyed.id] = -2
                    self.alive_agents -= 1

            # reward for distance to opps
            for ag_id, d in self.opp_to_escape.items():
                rew = 0
                if d and ag_id <= self.num_agents:
                    for d_opp in d:
                        if self.sim.unit_exists(ag_id) and self.sim.unit_exists(d_opp[0]):
                            d = self._distance(ag_id, d_opp[0])
                            if d < 0.06:
                                rew -= 0.02
                            elif d > 0.13:
                                rew += 0.02
                            # dynamic reward
                            # d = self._distance(ag_id, d_opp[0], True)
                            # rew += np.clip(0.1*(d-0.3), -0.03, 0.03) 
                    try:
                        rewards[ag_id] += rew
                    except:
                        pass

        return rewards

    def _get_policy(self):
        try:
            check_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'Level3_fight_2vs2', 'checkpoint')
            return Policy.from_checkpoint(check_path, ["ac1_policy", "ac2_policy"])
        except:
            raise NameError('Could not find L3 Fight Policy. Store in results/Level3_fight_2vs2/checkpoint')

    def _setup_sp_history(self):
        for i in range(1,6):
            if i == 5:
                ModelCatalog.register_custom_model(f"ac1_model",Esc1)
                ModelCatalog.register_custom_model(f"ac2_model",Esc2)
                try:
                    check_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'Level3_escape_2vs2', 'checkpoint')
                    self.sp_opps[i] = Policy.from_checkpoint(check_path, ["ac1_policy", "ac2_policy"])
                except:
                    raise NameError('Could not find L3 Esc Policy. Store in results/Level3_escape_2vs2/checkpoint')
            else:
                ModelCatalog.register_custom_model(f"ac1_model",Fight1)
                ModelCatalog.register_custom_model(f"ac2_model",Fight2)
                try:
                    check_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', f'Level{i}_fight_2vs2', 'checkpoint')
                    self.sp_opps[i] = Policy.from_checkpoint(check_path, ["ac1_policy", "ac2_policy"])
                except:
                    raise NameError(f'Could not find L{i} Fight Policy. Store in results/Level{i}_fight_2vs2/checkpoint')

    def _sp_actions(self, agend_id, unit):
        actions = {}
        
        def cc_obs(obs, esc=False):
            if unit.ac_type == 1:
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

        state = self._fight_state(agend_id) if self.opp_mode == "fight" else self._esc_state(agend_id)

        for ag_id, ag_state in state.items():
            if ag_id == agend_id:
                if self.curr_level == 4:
                    a = self.sp_opp[f'ac{ag_id-2}_policy'].compute_single_action(obs=cc_obs(ag_state), state=torch.zeros(1))
                else:
                    a = self.sp_opp[f'ac{ag_id-2}_policy'].compute_single_action(obs=cc_obs(ag_state, self.opp_mode=="escape"), state=torch.zeros(1))
                actions[ag_id] = a[0]

        return actions

    def _fight_state(self, agent_id=None):
        """
        Current observation, stored in state_dict. Destroyed agent's observation is filled with zeros, needed for Ray callback.
        """

        def fri_agent_state(act_agent_id, agent_type="agent"):
            if agent_type == "agent":
                oth_agent = 1 if act_agent_id == 2 else 2
            else:
                oth_agent = 3 if act_agent_id == 4 else 4
            if self.sim.unit_exists(oth_agent):
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
            else:
                return np.zeros(8)

        def opp_state(opp_id, agent_id, dist):
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

        state_dict = {}
        if agent_id:
            start = agent_id
            end = agent_id +1
        else:
            start = 1
            end = self.total_num +1

        for ag_id in range(start, end):
            state = []
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                d_opps = self._closest_object(ag_id)
                if d_opps:
                    unit = self.sim.get_unit(ag_id)
                    x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                    state.append(x)
                    state.append(y)
                    state.append(np.clip(unit.speed/unit.max_speed,0,1))
                    state.append(np.clip((unit.heading%359)/359, 0, 1))
                    state.append(self._focus_angle(ag_id, d_opps[0][0],True))
                    state.append(self._aspect_angle(d_opps[0][0], ag_id))
                    state.append(self._heading_diff(ag_id, d_opps[0][0]))
                    state.append(d_opps[0][1])
                    state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
                    if unit.ac_type == 1:
                        state.append(np.clip(unit.missile_remain/unit.rocket_max, 0, 1))
                        state.append(int(self.missile_wait[ag_id]==0))
                        state.append(int(bool(unit.actual_missile)))
                    else:
                        state.append(int(unit.cannon_current_burst_secs > 0))
                    state.extend(opp_state(d_opps[0][0], ag_id, d_opps[0][1]))
                    state.extend(fri_agent_state(ag_id, "agent" if ag_id<=self.num_agents else "opp"))
                    self.opp_to_attack[ag_id] = d_opps[0][0]
                else:
                    state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _esc_state(self, agent_id=None):
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
        
        def esc_fri_state(agent_id, agent_type):
            if agent_type == "agent":
                oth_agent = 1 if agent_id == 2 else 2
            else:
                oth_agent = 3 if agent_id == 4 else 4
            if self.sim.unit_exists(oth_agent):
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
            else:
                return np.zeros(8)
            
        state_dict = {}

        if agent_id:
            start = agent_id
            end = agent_id +1
        else:
            start = 1
            end = self.total_num +1

        for ag_id in range(start, end):
            state = []
            self.opp_to_attack[ag_id] = None
            self.opp_to_escape[ag_id] = None
            obs_dim = {1: 30, 2: 29, 3: 30, 4: 29}
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
                    if unit.ac_type==1:
                        state.append(np.clip(unit.missile_remain/(unit.rocket_max+1e-9), 0, 1))
                    opp_state = []
                    for d in d_opps:
                        opp_s = esc_opp_state(d[0], ag_id, d[1])
                        opp_state.extend(opp_s)
                    if len(opp_state) < 16:
                        opp_state.extend(np.zeros(16-len(opp_state)))
                    state.extend(opp_state)
                    state.extend(esc_fri_state(ag_id, "agent" if ag_id <=2 else "opp"))
                    self.opp_to_attack[ag_id] = d_opps[0][0]
                    self.opp_to_escape[ag_id] = d_opps
                else:
                    state = np.zeros(obs_dim[ag_id] if self.opp_mode=="escape" else self.obs_dim_map[ag_id], dtype=np.float32)
            else:
                state = np.zeros(obs_dim[ag_id] if self.opp_mode=="escape" else self.obs_dim_map[ag_id], dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)

        return state_dict

    def _closest_object(self, agent_id, once=False):
        """
        Return a sorted list with id's and distances to opponents. 
        """
        order = []
        if agent_id <= self.num_agents:
            start = self.num_agents + 1
            end = self.total_num + 1
        else:
            start = 1
            end = self.num_agents + 1
        for i in range(start, end):
            if self.sim.unit_exists(i):
                order.append([i, self._distance(agent_id, i, True)])
            if once:
                break
        order.sort(key=lambda x:x[1])
        return order

    def _focus_angle(self, agent_id, opp_id, norm=False):
        """
        Compute ATA angle based on vector angles of current heading direction and position of the two airplanes. 
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
        return self._shifted_range(d, 0, sqrt(0.18), 0, 1) if norm else d
        
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
        So, opponents shortly escape in the diagonal direction.
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
            if self.curr_level == 1:
                if r == 1:
                    x = random.uniform(7.12, 7.14)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                    a = random.randint(30, 150)
                elif r == 2:
                    x = random.uniform(7.16, 7.17)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                    a = random.randint(200, 330)
            elif self.curr_level == 2:
                if r == 1:
                    x = random.uniform(7.08, 7.13)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 180)
                elif r == 2:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(180, 359)
            elif self.curr_level >= 3:
                if r == 1:
                    x = random.uniform(7.07, 7.12)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(0, 270)
                elif r == 2:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(90, 359)

        elif agent == "opp":
            if self.curr_level == 1:
                if r == 1:
                    x = random.uniform(7.16, 7.17)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                elif r == 2:
                    x = random.uniform(7.12, 7.14)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
            elif self.curr_level == 2:
                if r == 1:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 359)
                elif r == 2:
                    x = random.uniform(7.08, 7.13)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 359)
            elif self.curr_level >= 3:
                if r == 1:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(0, 359)
                elif r == 2:
                    x = random.uniform(7.07, 7.12)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
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

            if i == 0:
                agent_unit = Rafale(Position(y, x, 10_000), heading=a, speed=100, group="agent")
            else:
                agent_unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=100, group="agent")

            self.sim.add_unit(agent_unit)
            self.sim.record_unit_trace(agent_unit.id)
            self.alive_agents += 1

        i = random.randint(0,1)
        x, y, a = self._sample_state("opp", i, r)

        opp_unit = Rafale(Position(y, x, 10_000), heading=a, speed=0 if self.curr_level < 2 else 200, group="opp")
        opp_unit.missile_remain = 10 if self.curr_level <= 4 else 5
        opp_unit.rocket_max = 10 if self.curr_level <= 4 else 5
        opp_unit.cannon_max = 400 if self.curr_level <= 4 else 300
        opp_unit.cannon_remain_secs = 400 if self.curr_level <= 4 else 300
        self.sim.add_unit(opp_unit)
        self.sim.record_unit_trace(opp_unit.id)
        self.alive_opps += 1

        i = 0 if i == 1 else 1
        x, y, a = self._sample_state("opp", i, r)

        opp_unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=0 if self.curr_level < 2 else 200, group="opp")
        opp_unit.cannon_max = 400 if self.curr_level <= 4 else 300
        opp_unit.cannon_remain_secs = 400 if self.curr_level <= 4 else 300
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
