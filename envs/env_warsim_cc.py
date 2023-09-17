"""
Environment for MARL Warsim, where 2 groups of airplanes fight against each other. 
"""

import time
import os
import random
import json
import numpy as np
from math import sin, cos, acos, pi, hypot, radians, exp, sqrt
from gym import spaces
from pathlib import Path
from typing import List, Dict, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from warsim.scenplotter.scenario_plotter import PlotConfig, ColorRGBA, StatusMessage, TopLeftMessage, \
    Airplane, PolyLine, Drawable, Waypoint, Missile, ScenarioPlotter # CORRECT THIS !! 

from warsim.simulator.cmano_simulator import Position, CmanoSimulator, UnitDestroyedEvent
from warsim.simulator.rafale_rocket import Rafale
from warsim.simulator.rafale_long import RafaleLong
from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits
from utils.angles import sum_angles

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from models.ac_models import CCRocket, CCEscape
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

colors = {
    'red_outline': ColorRGBA(0.8, 0.2, 0.2, 1),
    'red_fill': ColorRGBA(0.8, 0.2, 0.2, 0.2),
    'blue_outline': ColorRGBA(0.3, 0.6, 0.9, 1),
    'blue_fill': ColorRGBA(0.3, 0.6, 0.9, 0.2),
    'waypoint_outline': ColorRGBA(0.8, 0.8, 0.2, 1),
    'waypoint_fill': ColorRGBA(0.8, 0.8, 0.2, 0.2)
}
knots_to_ms = 0.514444 # used for predicted state

class BasicMarl(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True
        self.obs_dim = env_config.get("obs_dim", 0)

        self.observation_space = spaces.Box(low=np.zeros(self.obs_dim), high=np.ones(self.obs_dim), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([13,9,2,2])

    def reset(self):
        return np.zeros(self.obs_dim, dtype=np.float32)

    def step(self, action):
        return np.zeros(self.obs_dim, dtype=np.float32), 0, {"__all__": False}, {}
    
class EscapeMarl(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True

        self.observation_space = spaces.Box(low=np.zeros(15), high=np.ones(15), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([13,9])

    def reset(self):
        return np.zeros(15, dtype=np.float32)

    def step(self, action):
        return np.zeros(15, dtype=np.float32), 0, {"__all__": False}, {}

#class DogfightScenario(MultiAgentEnv, TaskSettableEnv):
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
        
        self.ss = env_config.get("ss", 1)
        self.obs_dim = env_config.get("obs_dim", 23)
        self.horizon = env_config.get("horizon",200)
        self.curr_level = env_config.get("level",1)
        self.restore_path = env_config.get("restore_path", None)
        self.log_path = env_config.get("log_path", None)
        self.agent_mode = env_config.get("agent_mode", "fight")
        self.opp_mode = env_config.get("opp_mode", "fight")
        self.ac_types = env_config.get("ac_type", 1)
        
        self.steps = 0
        self.opp_to_attack = {}
        self.missile_wait = {}
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.firendly_kills = 0
        self.opp_kills = 0
        self.got_killed = 0
        self.sp_update = False
        self.horizon_level = {1: 150, 2:200, 3:300, 4:300, 5:400}

        self.observation_space = spaces.Box(low=np.zeros(self.obs_dim), high=np.ones(self.obs_dim), dtype=np.float32)
        if self.ac_types == 1:
            self.action_space= spaces.MultiDiscrete([13,9,2,2])
        else:
            self.action_space= spaces.MultiDiscrete([13,9,2])
        self._agent_ids = set(range(1,self.total_num+1))

        # Plotting
        self.plt_cfg = PlotConfig()
        self.plt_cfg.units_scale = 20.0
        self.plotter = ScenarioPlotter(self.map_limits, dpi=200, config=self.plt_cfg)

        self.sp_opp = None
        if self.curr_level >= 4:
           self.sp_opp = self._setup_sp_opponent(True)

    def reset(self):
        if self.args.persistence:
            self._update_persistence()
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.steps = 0
        self.alive_agents = 0
        self.alive_opps = 0
        self.missile_wait = {i:0 for i in range(1, self.total_num+1)}
        self.opp_to_attack = {i:None for i in range(1, self.total_num+1)}
        self.sim = CmanoSimulator(num_units=self.num_agents, num_opp_units=self.num_opps)
        self._reset_scenario()
        if self.curr_level >=4 and os.listdir(os.path.join(self.log_path, 'checkpoint')) and self.sp_update:
            if self.opp_mode == "fight":
                self.sp_opp = self._setup_sp_opponent()
                self.sp_opp.restore(os.path.join(self.log_path, 'checkpoint'))
            else:
                self.sp_opp = self._setup_escape_opp()
            self.sp_update = False
        self.horizon = self.horizon_level[self.curr_level]
        return self._state_cc()

    def step(self, action):
        """
        Forward current action to sub-env, update sim. 
        """
        self.steps += 1
        reward = {}
        opp_actions = {}
        if action:
            reward, opp_actions = self._take_action_get_reward(action)
        done = self.alive_agents <= 0 or self.alive_opps <= 0 or self.steps >= self.horizon
        return self._state_cc(), reward, {"__all__": done}, {1:opp_actions, 2:opp_actions}

    def _take_action_get_reward(self, action):
        """
        Apply actions to agents and opponents and directly compute rewards.
        """
        rewards = {}
        opp_stats = {}
        opp_actions = {3:[0,0,0,0], 4:[0,0,0,0]}

        for i in range(1, self.total_num+1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if i <= self.num_agents or self.curr_level >= 4:
                    if i >= self.num_agents+1:
                        actions = self._sp_actions() if self.opp_mode == "fight" else self._esc_actions()
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
                            rewards[i] = -0.05
                    if self.ac_types == 1:
                        if bool(actions[i][3]) and u.ac_type == 1 and u.missile_remain > 0 and self.opp_to_attack[i] and not u.actual_missile and self.missile_wait[i] == 0:
                            u.fire_missile(u, self.sim.get_unit(self.opp_to_attack[i]), self.sim)
                            if i <= self.num_agents:
                                self.missile_wait[i] = random.randint(10,25)
                                if self.agent_mode == "escape" and u.missile_remain < 3:
                                    rewards[i] = -0.05
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
                        if self.steps==1 or self.steps % random.randint(35,45) <= 5:
                            r = random.randint(0,1)
                            h = 12 if r==0 else 0
                            u.set_heading((u.heading + ((-1)**r)*90)%360) #12
                            s = 100 + random.randint(0,4)*75
                            u.set_speed(s) # int(s/100)
                            opp_actions[i] = [h, int(s/100), 1, 0]
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
                #if isinstance(ev, UnitDestroyedEvent):
                if self.ss == 1:
                    if ev.unit_killer.id <= self.num_agents and ev.unit_destroyed.id in range(self.num_agents+1, self.total_num+1):

                        if ev.origin.id >= self.total_num+1 and self.ac_types==1: # rocket
                            #rewards[ev.unit_killer.id] = np.clip(ev.unit_killer.missile_remain/(ev.unit_killer.rocket_max+1e-9)+0.5,0.5,1.5)
                            #rewards[ev.unit_killer.id] = 1
                            rewards[ev.unit_killer.id] = self._shifted_range(ev.unit_killer.missile_remain/ev.unit_killer.rocket_max, 0,1, 1,1.5)
                        else:
                            #rewards[ev.unit_killer.id] = np.clip(ev.unit_killer.cannon_remain_secs/ev.unit_killer.cannon_max + opp_stats[ev.unit_killer.id][0], 0.5, 2)
                            #rewards[ev.unit_killer.id] = 1.5 * self._shifted_range(opp_stats[ev.unit_killer.id][0], 0,1, 0.5,1)
                            rewards[ev.unit_killer.id] = self._shifted_range(ev.unit_killer.cannon_remain_secs/ev.unit_killer.cannon_max, 0,1, 0.5,1) + self._shifted_range(opp_stats[ev.unit_killer.id][0], 0,1, 0.5,1) #range [1,2]

                        self.alive_opps -= 1
                        self.opp_kills += 1

                    # killed by opp
                    elif ev.unit_destroyed.id <= self.num_agents and ev.unit_killer.id in range(self.num_agents+1, self.total_num+1):
                        rewards[ev.unit_destroyed.id] = -2
                        self.alive_agents -= 1
                        self.got_killed += 1
                else:
                    if ev.unit_killer.id <= self.num_agents:
                        if ev.unit_destroyed.id in range(self.num_agents+1, self.total_num+1):

                            if ev.origin.id >= self.total_num+1 and self.ac_types==1: # rocket
                                #rewards[ev.unit_killer.id] = np.clip(ev.unit_killer.missile_remain/(ev.unit_killer.rocket_max+1e-9)+0.5,0.5,1.5)
                                #rewards[ev.unit_killer.id] = 1
                                rewards[ev.unit_killer.id] = self._shifted_range(ev.unit_killer.missile_remain/ev.unit_killer.rocket_max, 0,1, 1,1.5)
                            else:
                                #rewards[ev.unit_killer.id] = np.clip(ev.unit_killer.cannon_remain_secs/ev.unit_killer.cannon_max + opp_stats[ev.unit_killer.id][0], 0.5, 2)
                                #rewards[ev.unit_killer.id] = 1.5 * self._shifted_range(opp_stats[ev.unit_killer.id][0], 0,1, 0.5,1)
                                rewards[ev.unit_killer.id] = self._shifted_range(ev.unit_killer.cannon_remain_secs/ev.unit_killer.cannon_max, 0,1, 0.5,1) + self._shifted_range(opp_stats[ev.unit_killer.id][0], 0,1, 0.5,1) #range [1,2]

                            self.alive_opps -= 1
                            self.opp_kills += 1

                        # killed by friend
                        elif ev.unit_destroyed.id <= self.num_agents:
                            rewards[ev.unit_killer.id] = -2
                            self.alive_agents -= 1
                            self.firendly_kills += 1

                    # killed by opp
                    elif ev.unit_destroyed.id <= self.num_agents and ev.unit_killer.id in range(self.num_agents+1, self.total_num+1):
                        rewards[ev.unit_destroyed.id] = -2
                        self.alive_agents -= 1
                        self.got_killed += 1

        else:
            for ev in events:
                if ev.unit_killer.id <= self.num_agents:
                    if ev.unit_destroyed.id in range(self.num_agents+1, self.total_num+1):

                        rewards[ev.unit_killer.id] = 1
                        self.alive_opps -= 1
                        self.opp_kills += 1

                    elif ev.unit_destroyed.id <= self.num_agents:
                        rewards[ev.unit_killer.id] = -1
                        self.alive_agents -= 1
                        self.firendly_kills += 1

                elif ev.unit_destroyed.id <= self.num_agents and ev.unit_killer.id in range(self.num_agents+1, self.total_num+1):
                    rewards[ev.unit_destroyed.id] = -1
                    self.alive_agents -= 1
                    self.got_killed += 1

            for i in rewards.keys():
                rew = 0
                for d_i in range(self.num_agents+1, self.total_num+1):
                    if self.sim.unit_exists(i) and self.sim.unit_exists(d_i):
                        d = self._distance(i, d_i)
                        if d < 0.06:
                            rew -= 0.02
                        elif d > 0.14:
                            rew += 0.02
                    rewards[i] += rew

        return rewards      

    def _update_persistence(self):
        data = None
        while not data:
            try:
                with open(os.path.join(self.log_path, "metrics.json"), "r") as file:
                    data = json.load(file)
            except:
                pass

        data["friendly_kills"] += self.firendly_kills
        data["opponent_kills"] += self.opp_kills
        data["got_killed"] += self.got_killed

        self.curr_level = data["level"]
        self.sp_update = data["sp_update"]

        data["sp_update"] = False

        self.opp_mode = "fight" if data["escape_wait"] > 0 else "escape"

        written = False
        while not written:
            try:
                with open(os.path.join(self.log_path, "metrics.json"), "w") as file:
                    json.dump(data, file)
                written = True
            except:
                pass

        self.firendly_kills = 0
        self.opp_kills = 0
        self.got_killed = 0
        
    def _setup_sp_opponent(self, restore=False):
        class FillInActions(DefaultCallbacks):
            def on_postprocess_trajectory(
                self,
                worker,
                episode,
                agent_id,
                policy_id,
                policies,
                postprocessed_batch,
                original_batches,
                **kwargs
            ):
                to_update = postprocessed_batch[SampleBatch.CUR_OBS]
                other_id = 2 if agent_id == 1 else 1
                _, opponent_batch = original_batches[other_id]
                try:
                    opponent_action = np.squeeze(opponent_batch[SampleBatch.ACTIONS])
                except:
                    opponent_action = None
                if opponent_action is not None:
                    if opponent_action.ndim == 1:
                        a = round(float(opponent_action[0] / 12.0), 3)
                        b = round(float(opponent_action[1] / 8.0), 3)
                        c = round(float(opponent_action[2]), 3)
                        d = round(float(opponent_action[3]), 3)
                        to_update[:, :4] = np.array([a,b,c,d])
                    else:
                        a = opponent_action[:,0]
                        a_val = np.zeros((len(a)), dtype=np.float32)
                        for i,v in enumerate(a):
                            a_val[i] = round(float(v)/12.0,3)
                        b = opponent_action[:,1]
                        b_val = np.zeros((len(b)), dtype=np.float32)
                        for i,v in enumerate(b):
                            b_val[i] = round(float(v/8),3)
                        c = opponent_action[:,2]
                        d = opponent_action[:,3]
                        to_update[:,0] = a_val
                        to_update[:,1] = b_val
                        to_update[:,2] = c
                        to_update[:,3] = d
        
        def central_critic_observer(agent_obs, **kw):
            new_obs = {
                3: {
                    "own_obs": agent_obs[3] , #if 1 in agent_obs else np.zeros(11, dtype=np.float32),
                    "opponent_obs": agent_obs[4], #if 2 in agent_obs else np.zeros(11, dtype=np.float32),
                    "opponent_action": np.zeros(4),  # filled in by FillInActions
                },
                4: {
                    "own_obs": agent_obs[4], #if 2 in agent_obs else np.zeros(11, dtype=np.float32),
                    "opponent_obs": agent_obs[3], #if 1 in agent_obs else np.zeros(11, dtype=np.float32),
                    "opponent_action": np.zeros(4),  # filled in by FillInActions
                },
            }
            return new_obs

        observer_space = spaces.Dict(
            {
                "own_obs": spaces.Box(low=0, high=1, shape=(self.obs_dim,)),
                "opponent_obs": spaces.Box(low=0, high=1, shape=(self.obs_dim,)),
                "opponent_action": spaces.Box(low=0, high=12, shape=(4,), dtype=np.float32),
            }
        )

        ModelCatalog.register_custom_model("fight_model",CCRocket)
        action_space = spaces.MultiDiscrete([13,9,2,2])

        algo = (
            PPOConfig()
            .rollouts(num_rollout_workers=0, horizon=200, batch_mode="complete_episodes")
            .resources(num_gpus=0)
            .evaluation(evaluation_interval=None)
            .environment(env=BasicMarl, env_config={"obs_dim":self.obs_dim})
            .training(train_batch_size=2000, gamma=0.99, clip_param=0.25,lr=1e-4, lambda_=0.95)
            .framework("torch")
            .exploration(explore=False)
            .multi_agent(policies={
                    "fight_policy": PolicySpec(
                        None,
                        observer_space,
                        action_space,
                        config={
                            "model": {
                                "custom_model": "fight_model"
                            }
                        }
                    )
                },
                policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "fight_policy"),
                observation_fn=central_critic_observer)
            .callbacks(FillInActions)
            .build()
        )
        if restore and self.args.restore:
            algo.restore(self.restore_path)
        self.opp_mode = "fight"
        return algo

    def _setup_escape_opp(self):
        ModelCatalog.register_custom_model("escape_model", CCEscape)
        algo = (
            PPOConfig()
            .rollouts(num_rollout_workers=0, horizon=200, batch_mode="complete_episodes")
            .resources(num_gpus=0)
            .evaluation(evaluation_interval=None)
            .environment(env=EscapeMarl, env_config={})
            .training(train_batch_size=2000, gamma=0.99, clip_param=0.25,lr=1e-4, lambda_=0.95)
            .framework("torch")
            .exploration(explore=False)
            .multi_agent(policies={
                    "escape_policy": PolicySpec(
                        config={
                            "model": {
                                "custom_model": "escape_model"
                            }
                        }
                    )
                },
                policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "escape_policy"))
            .build()
        )
        algo.restore('/home/ardianselmonaj/Projects/marl-warsim/results/escape/escape_lv3/checkpoint')
        self.opp_mode = "escape"
        return algo

    def _sp_actions(self):
        def cc_obs(obs, id):
            if id == 3:
                return {
                    "own_obs": obs[3],
                    "opponent_obs": obs[4],
                    "opponent_action": np.zeros(4)
                }
            elif id == 4:
                return {
                    "own_obs": obs[4],
                    "opponent_obs": obs[3],
                    "opponent_action": np.zeros(4)
                }
        if not self.sp_opp:
            if self.opp_mode == "fight":
                self.sp_opp = self._setup_sp_opponent()
                self.sp_opp.restore(os.path.join(self.log_path, 'checkpoint'))
            else:
                self.sp_opp = self._setup_escape_opp()

        state = self._state("opp")
        actions = {}
        for ag_id, ag_state in state.items():
            actions[ag_id] = self.sp_opp.compute_single_action(observation=cc_obs(state, ag_id), policy_id="fight_policy")
            
        return actions

    def _esc_actions(self):
        actions = {}
        state = self._esc_state("opp")
        for opp_id, opp_state in state.items():
            if self.sim.unit_exists(opp_id):
                actions[opp_id] = self.sp_opp.compute_single_action(observation=opp_state, policy_id="escape_policy")

        return actions

    def _state(self, agent_type="agent"):
        """
        Current observation, stored in state_dict, only containing alive agents.
        """
        state_dict = {}
        if agent_type == "agent":
            start = 1
            end = self.num_agents +1
        else:
            start = self.num_agents + 1
            end = self.total_num +1

        for ag_id in range(start, end):
            state = []
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                d_opps = self._closest_object(ag_id)
                if d_opps:
                    unit = self.sim.get_unit(ag_id)
                    x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                    if self.args.heterogeneous:
                        state.append(unit.ac_type/2)
                    state.append(x)
                    state.append(y)
                    state.append(np.clip((unit.heading%359)/359, 0, 1))
                    state.append(unit.speed/unit.max_speed)
                    state.append(np.clip(self._focus_angle(ag_id, d_opps[0][0]) / 180, 0, 1))
                    state.append(d_opps[0][1])
                    state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
                    if self.ac_types == 1:
                        state.append(np.clip(unit.missile_remain/(unit.rocket_max+1e-9), 0, 1))
                        state.append(int(self.missile_wait[ag_id]==0))
                        state.extend(self._missile_state(unit))
                    # if self.ss >= 2:
                    #     state.extend(self._oth_agent_state(ag_id, agent_type))
                    state.extend(self._opp_state(d_opps[0][0], ag_id))
                    self.opp_to_attack[ag_id] = d_opps[0][0]
                    if self.ss >= 2:
                        state.extend(self._extended_state())
                else:
                    state = np.zeros(self.obs_dim, dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim, dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _state_cc(self,):
        """
        Current observation, stored in state_dict, only containing alive agents.
        """
        state_dict = {}

        for ag_id in range(1, self.total_num+1):
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
                    if self.ac_types == 1:
                        state.append(np.clip(unit.missile_remain/unit.rocket_max, 0, 1))
                        state.append(int(self.missile_wait[ag_id]==0))
                        state.append(int(bool(unit.actual_missile)))
                    state.extend(self._opp_state(d_opps[0][0], ag_id, d_opps[0][1]))
                    if self.ss >= 2:
                        state.extend(self._oth_agent_state(ag_id, "agent" if ag_id<=2 else "opp"))
                    self.opp_to_attack[ag_id] = d_opps[0][0]
                else:
                    state = np.zeros(self.obs_dim, dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim, dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _esc_state(self, agent_type="agent"):
        """
        Observation for escaping.
        """
        state_dict = {}
        rocket_ids = []
        if agent_type == "agent":
            start = 1
            end = self.num_agents +1
        else:
            start = self.num_agents + 1
            end = self.total_num +1

        for ag_id in range(start, end):
            state = []
            if self.sim.unit_exists(ag_id):
                unit = self.sim.get_unit(ag_id)
                x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                if self.args.heterogeneous:
                    state.append(unit.ac_type/2)
                state.append(x)
                state.append(y)
                state.append(np.clip((unit.heading%359)/359, 0, 1))
                state.append(unit.speed/unit.max_speed)
                state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
                if self.ac_types == 1:
                    state.append(np.clip(unit.missile_remain/(unit.rocket_max+1e-9), 0, 1))
                    state.append(int(self.missile_wait[ag_id]==0))
                    #state.extend(self._missile_state(unit, False))

                k_state = []
                k_list = [i for i in range(1,self.total_num+1)] #attention when more agents and opps
                k_list.remove(ag_id)
                for k in k_list:
                    if self.sim.unit_exists(k):
                        k_unit = self.sim.get_unit(k)
                        x_k,y_k = self.map_limits.relative_position(k_unit.position.lat, k_unit.position.lon)
                        k_state.append(k_unit.ac_type/2)
                        k_state.append(x_k)
                        k_state.append(y_k)
                        k_state.append(np.clip((k_unit.heading%359)/359, 0, 1))
                        k_state.append(self._distance(ag_id, k))
                        #k_state.extend(self._missile_state(k_unit, False))
                        if k_unit.actual_missile:
                            rocket_ids.append(k_unit.actual_missile.id)
                    else:
                        k_state.extend(np.zeros(5))

                state.extend(k_state)

                closest_rockets = self._closest_rockets(ag_id, rocket_ids)
                r_list = []
                for r in closest_rockets:
                    r_unit = self.sim.get_unit(r[0])
                    x_r,y_r = self.map_limits.relative_position(r_unit.position.lat, r_unit.position.lon)
                    r_list.append(x_r)
                    r_list.append(y_r)
                    if len(r_list) == 4:
                        break
                
                state.extend(r_list)
                state.extend(np.zeros(4-len(r_list)))

            else:
                state = np.zeros(self.obs_dim)

            state_dict[ag_id] = np.array(state, dtype=np.float32)

        return state_dict

    def _oth_agent_state(self, act_agent_id, agent_type="agent"):
        if agent_type == "agent":
            oth_agent = 1 if act_agent_id == 2 else 2
        else:
            oth_agent = 3 if act_agent_id == 4 else 4
        if self.sim.unit_exists(oth_agent):
            unit = self.sim.get_unit(oth_agent)
            state = []
            #state.append(unit.ac_type/2)
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(self._heading_diff(act_agent_id, oth_agent))
            state.append(self._focus_angle(act_agent_id, oth_agent, True))
            state.append(self._focus_angle(oth_agent, act_agent_id, True))
            state.append(self._distance(act_agent_id, oth_agent, True))
            state.append(int(unit.cannon_current_burst_secs > 0))
            return state
        else:
            return np.zeros(7)
            
    def _extended_state(self):
        """
        observations for messages to policy.
        """
        state = []
        iter_list = []
        if self.ss == 2 :
            iter_list = [i for i in range(1,self.num_agents+1)]
        if self.ss == 2.1 :
            iter_list = [i for i in range(1,self.total_num+1)]
        elif self.ss == 3: 
            iter_list = [i for i in range(1,self.total_num+1)]
        elif self.ss == 4:
            iter_list = [i for i in range(1,self.num_agents+1)]
            iter_list.extend([i for i in range(1,self.total_num+1)])

        for i in iter_list:
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                #state.append(unit.ac_type/2)
                state.append(x)
                state.append(y)
                state.append(np.clip((unit.heading%359)/359, 0, 1))
                state.append(unit.speed/unit.max_speed)
                if self.ss <= 3:
                    state.extend(self._predicted_state(unit))
            else:
                if self.ss <= 3:
                    state.extend(np.zeros(6))
                else:
                    state.extend(np.zeros(4))

        return state

    def _predicted_state(self, unit):
        """
        Get a prediction of future position with some randomness.
        """
        pred = np.zeros(2)
        tick_error = random.uniform(10, 20)
        d = geodetic_direct(unit.position.lat, unit.position.lon, unit.heading, unit.speed * knots_to_ms * tick_error)
        x, y = self.map_limits.relative_position(d[0], d[1])
        pred[0] = x
        pred[1] = y
        return pred

    def _missile_state(self, unit, head=True):
        if unit.actual_missile:
            state = []
            x, y = self.map_limits.relative_position(unit.actual_missile.position.lat, unit.actual_missile.position.lon)
            state.append(x)
            state.append(y)
            if head:
                state.append(np.clip((unit.actual_missile.heading%359)/359, 0, 1))
            return state
        else:
            if head:
                return np.zeros(3)
            else:
                return np.zeros(2)
            
    def _opp_state(self, opp_id, agent_id, dist):
        """
        Depending on state-space option, compute the opponent part of the whole observation. 
        """
        state = []
        unit = self.sim.get_unit(opp_id)
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        #state.append(unit.ac_type/2)
        state.append(x)
        state.append(y)
        state.append(np.clip(unit.speed/unit.max_speed,0,1))
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        state.append(self._focus_angle(opp_id, agent_id,True))
        state.append(self._aspect_angle(agent_id, opp_id))
        state.append(self._heading_diff(opp_id, agent_id))
        state.append(dist)
        state.append(int(unit.cannon_current_burst_secs > 0))
        state.append(int(bool(unit.actual_missile)))
        return state

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

    def _closest_rockets(self, ag_id, rocket_ids):
        dists = []
        for rid in rocket_ids:
            if self.sim.unit_exists(rid):
                dists.append([rid, self._distance(ag_id, rid)])
        dists.sort(key=lambda x:x[1])
        return dists

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
        r = random.randint(1,2)
        for i in range(self.num_agents):
            x, y, a = self._sample_state("agent", i, r)

            if self.args.heterogeneous:
                if random.randint(0,1):
                    agent_unit = Rafale(Position(y, x, 10_000), heading=a, speed=100, group="agent", friendly_check=self.ss>=2 or self.agent_mode=="escape")
                else:
                    agent_unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=100, group="agent", friendly_check=self.ss>=2 or self.agent_mode=="escape")
            else:
                if self.ac_types == 1:
                    agent_unit = Rafale(Position(y, x, 10_000), heading=a, speed=100, group="agent", friendly_check=self.ss>=2 or self.agent_mode=="escape")
                else:
                    agent_unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=100, group="agent", friendly_check=self.ss>=2 or self.agent_mode=="escape")

            self.sim.add_unit(agent_unit)
            self.sim.record_unit_trace(agent_unit.id)
            self.alive_agents += 1

        for i in range(self.num_opps):
            x, y, a = self._sample_state("opp", i, r)

            #if self.args.heterogeneous:
            #if random.randint(0,1):
            opp_unit = Rafale(Position(y, x, 10_000), heading=a, speed=0 if self.curr_level < 2 else 200, group="opp", friendly_check=self.ss>=2 or self.opp_mode=="escape")
            opp_unit.missile_remain = 10
            opp_unit.rocket_max = 10
            #else:
                #opp_unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=0 if self.curr_level < 2 else 200, group="opp", friendly_check=self.ss>=2 or self.opp_mode=="escape")
            # else:
            #     opp_unit = Rafale(Position(y, x, 10_000), heading=a, speed=100, group="agent", friendly_check=self.ss>=2 or self.agent_mode=="escape")
            #     opp_unit.missile_remain = 10
            #     opp_unit.rocket_max = 10

            opp_unit.cannon_max = 400
            opp_unit.cannon_remain_secs = 400
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
