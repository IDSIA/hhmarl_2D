import gym
import random
import numpy as np
from math import sin, cos, acos, pi, hypot, radians, floor
from gym import spaces
from pathlib import Path
from typing import List, Dict, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from backup_files.nn_models.shp_mod import ShPModel
from models.escape_mod import EscapeOne
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from warsim.scenplotter.scenario_plotter import PlotConfig, ScenarioPlotter, ColorRGBA, StatusMessage, TopLeftMessage, \
    Airplane, PolyLine, Drawable, Waypoint

from warsim.simulator.cmano_simulator import Position, CmanoSimulator
from warsim.simulator.rafale_unit import Rafale
from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits
from utils.angles import sum_angles

colors = {
    'red_outline': ColorRGBA(0.8, 0.2, 0.2, 1),
    'red_fill': ColorRGBA(0.8, 0.2, 0.2, 0.2),
    'blue_outline': ColorRGBA(0.3, 0.6, 0.9, 1),
    'blue_fill': ColorRGBA(0.3, 0.6, 0.9, 0.2),
    'waypoint_outline': ColorRGBA(0.8, 0.8, 0.2, 1),
    'waypoint_fill': ColorRGBA(0.8, 0.8, 0.2, 0.2)
}

class BasicMarl(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True

        self.observation_space = spaces.Box(low=np.zeros(11), high=np.ones(11), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([13,9,2])

    def reset(self):
        return np.zeros(11, dtype=np.float32)

    def step(self, action):
        return np.zeros(11, dtype=np.float32), 0, {"__all__": False}, {}

class BasicEscape(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True

        self.observation_space = spaces.Box(low=np.zeros(15), high=np.ones(15), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([13,9])

    def reset(self):
        return np.zeros(15, dtype=np.float32)

    def step(self, action):
        return np.zeros(15, dtype=np.float32), 0, {"__all__": False}, {}


class DogfightHierarchy(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True

        self.sim: Optional[CmanoSimulator] = None
        self.map_limits = MapLimits(7.0, 5.0, 7.5, 5.5)

        self.env_config = env_config
        self.args = env_config.get("args", None)
        self.mode = env_config.get("mode", None)
        self.n_sub_steps = env_config.get("sub_steps", 20)
        self.num_opps = env_config.get("num_opps",2)
        self.num_agents = env_config.get("num_agents",2)
        self.total_num = self.num_agents + self.num_opps
        self.alive_agents = 0
        self.alive_opps = 0
        
        self.policy = env_config.get("policy","fight")
        self.horizon = env_config.get("horizon",200)
        self.curr_level = env_config.get("level",1)
        self.ss = env_config.get("ss",1)
        self.path_fight = env_config.get("path_fight", None)
        self.path_escape = env_config.get("path_escape",None)
        
        self.steps = 0
        self.fire_max = self.horizon - 100
        self.opponent_firing = {i:False for i in range(self.num_agents+1, self.num_agents+self.num_opps+1)}

        self.observation_space = spaces.Box(low=np.ones(93)*-10, high=np.ones(93)*10, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([6,6,6,6,6])

        # Plotting
        self.plt_cfg = PlotConfig()
        self.plt_cfg.units_scale = 20.0
        self.plotter = ScenarioPlotter(self.map_limits, dpi=200, config=self.plt_cfg)

        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0

        self.policy_fight = self.setup_fight()
        self.policy_escape = self.setup_escape()

    def setup_fight(self):
        #ModelCatalog.register_custom_model("shp_model",ShPModel)
        algo = (
            PPOConfig()
            .rollouts(num_rollout_workers=4, horizon=400, batch_mode="complete_episodes")
            .resources(num_gpus=0)
            .evaluation(evaluation_interval=None)
            .environment(env=BasicMarl, env_config={})
            .training(train_batch_size=2000, gamma=0.99, clip_param=0.2,lr=1e-4, lambda_=0.8)
            .framework("torch")
            .exploration(explore=False)
            .multi_agent(policies={
                    "shared_policy": PolicySpec(
                        config={
                            "model": {
                                "fcnet_hiddens" : [512, 512],
                                "fcnet_activation": "tanh",
                                "vf_share_layers": False,
                            }
                        }
                    )
                },
                policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "shared_policy"))
            .build()
        )
        algo.restore(self.path_fight)
        return algo

    def setup_escape(self):
        ModelCatalog.register_custom_model("escape_model", EscapeOne)
        algo = (
            PPOConfig()
            .rollouts(num_rollout_workers=4, horizon=400, batch_mode="complete_episodes")
            .resources(num_gpus=0)
            .evaluation(evaluation_interval=None)
            .environment(env=BasicEscape, env_config={})
            .training(train_batch_size=2000, gamma=0.99, clip_param=0.2,lr=1e-4, lambda_=0.8)
            .framework("torch")
            .exploration(explore=False)
            .multi_agent(policies={
                    "shared_policy": PolicySpec(
                        config={
                            "model": {
                                "custom_model": "escape_model"
                            }
                        }
                    )
                },
                policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "shared_policy"))
            .build()
        )
        algo.restore(self.path_escape)
        return algo

    def reset(self):
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.steps = 0
        self.alive_agents = 0
        self.alive_opps = 0
        self.num_agents = random.randint(2,5)
        self.num_opps = random.randint(2,5)
        self.total_num = self.num_agents + self.num_opps
        self.opponent_firing = {i:False for i in range(self.num_agents+1, self.num_agents+self.num_opps+1)}
        self.sim = CmanoSimulator()
        self._reset_scenario()
        return self._state()

    def _fight_step(self, agent_id, opp_id):
        def dist(agent_id, opp_id):
            return hypot(self.sim.get_unit(opp_id).position.lon - self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat - self.sim.get_unit(agent_id).position.lat)
        
        def focus(agent_id, opp_id):
            return np.clip(self._focus_angle(agent_id, opp_id) / 180, 0, 1)

        state = []

        unit = self.sim.get_unit(agent_id)
        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.append(x)
        state.append(y)
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        state.append(focus(agent_id, opp_id))
        state.append(dist(agent_id, opp_id))
        state.append(np.clip(unit.cannon_remain_secs/self.fire_max, 0, 1))

        unit = self.sim.get_unit(opp_id)
        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.append(x)
        state.append(y)
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        state.append(focus(opp_id, agent_id))
        state.append(int(self.opponent_firing[opp_id]))

        if len(state) < 11:
            state.extend(np.zeros(11-len(state)))

        return self.policy_fight.compute_single_action(observation=np.array(state, dtype=np.float32), policy_id="shared_policy")

    def _escape_step(self, agent_id):
        state = []
        unit = self.sim.get_unit(agent_id)
        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.append(x)
        state.append(y)
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        remaining = self._all_distances(agent_id, with_af_id=True)
        if remaining:
            for k, item in enumerate(remaining):
                x_op, y_op = self.map_limits.relative_position(self.sim.get_unit(item[0]).position.lat, self.sim.get_unit(item[0]).position.lon)
                d = item[1]
                af = item[2]
                state.append(x_op)
                state.append(y_op)
                state.append(af)
                state.append(d)

                if k == 2:
                    break
        else:
            state.extend(np.zeros(12))

        if len(state) < 15:
            state.extend(np.zeros(15-len(state)))

        return self.policy_escape.compute_single_action(observation=np.array(state, dtype=np.float32), policy_id="shared_policy")
        
    def _sub_steps(self, actions):
        """
        This method iteratively calls fight_step and / or escape_steps, s.th. both actions are done simultaneously. 
        This will run for fixed time steps (e.g. 30) or when an event happend (kill, outside bounds)
        """
        s = 0
        reward = 0
        event = False
        while s <= self.n_sub_steps and not event:
            actions_per_agent = {}
            for i, act in enumerate(actions, start=1):
                #act = act + self.num_agents # shift
                if not self.sim.unit_exists(act+self.num_agents):
                    reward -= 2 # punish network when chosing not existing opponents

                else:
                    if i <= self.num_agents:
                        if self.sim.unit_exists(i):
                            if act == 0:
                                actions_per_agent[i] = self._escape_step(i)
                            else:
                                actions_per_agent[i] = self._fight_step(i, act+self.num_agents)
                
            self._take_sub_action(actions_per_agent)
            reward, event = self._get_rewards(actions)
            s += 1
            self.steps += 1
        return reward

    def _get_rewards(self, global_actions):
        _, kills = self.sim.do_tick()
        rewards = 0
        event = False
        for i in range(1, self.total_num+1):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)

                if not self.map_limits.in_boundary(unit.position.lat, unit.position.lon):
                    self.sim.remove_unit(i)
                    if i <= self.num_agents:
                        rewards -= 5
                        self.alive_agents -= 1
                    else:
                        self.alive_opps -= 1

                else:
                    if i <= self.num_agents:
                        if global_actions[i-1] == 0:
                            dists = self._all_distances(i)
                            for k, d in enumerate(dists):
                                if d < 0.07:
                                    rewards -= 0.02
                                elif d > 0.2:
                                    rewards += 0.02
                                if k == 2: # the 3 closest objects only
                                    break
                        else:
                            if i in list(kills.keys()):
                                rewards += max(min(3, 3*((self.horizon-self.steps)/self.horizon)), 1.5)
                                self.alive_opps -= 1
                                event = True
                            elif i in list(kills.values()):
                                rewards -= 2
                                self.alive_agents -= 1
                                event = True
        return rewards, event

    def _take_sub_action(self, actions):

        for i in range(1, self.total_num+1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if i <= self.num_agents:
                    try:
                        u.set_heading((u.heading + (actions[i][0]-6)*15)%360)
                        u.set_speed(100+100*actions[i][1])
                        if bool(actions[i][2]) and u.cannon_remain_secs > 0:
                            u.fire_cannon()
                    except:
                        pass

                else:
                    if self.steps % 60 == 0 and not self.hardcoded_opps_escaping:
                        self.hardcoded_opps_escaping = bool(random.randint(0, 1))
                        if self.hardcoded_opps_escaping:
                            self.opps_escaping_time = int(random.uniform(25, 35))

                    if self.hardcoded_opps_escaping:
                        heading, speed, fire = self._escaping_opps(u)
                        self.opps_escaping_time -= 1
                        if self.opps_escaping_time <= 0:
                            self.hardcoded_opps_escaping = False
                    else:
                        heading, speed, fire = self._hardcoded_opp(i)
                    u.set_heading(heading)
                    u.set_speed(speed)
                    self.opponent_firing[i] = fire
                    if fire:
                        u.fire_cannon()

    def step(self, actions):
        #self.steps += 1
        #print("global action ", self.steps)
        reward = self._sub_steps(actions)
        done = self.alive_agents == 0 or self.alive_opps == 0 or self.steps >= self.horizon
        return self._state(), reward, done, {}

    def _state(self):
        state = []
        
        for i in range(1, self.num_agents+1):
            unit_state = []
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                unit_state.append(x)
                unit_state.append(y)
                unit_state.append(np.clip((unit.heading%359)/359, 0, 1))
                unit_state.append(np.clip(unit.cannon_remain_secs/self.fire_max, 0, 1))
                d_opps = self._closest_objects(i)
                for item in d_opps:
                    if item[0] is not None:
                        unit_state.append(item[2])
                        unit_state.append(item[1])
                    else:
                        unit_state.extend(np.zeros(2))
                if len(unit_state) != 14:
                    unit_state.extend(np.zeros(14-len(unit_state)))
                state.extend(unit_state)
            else:
                state.extend(np.zeros(14))

        if len(state) < 70:
            state.extend(np.zeros(70-len(state)))

        for i in range(self.num_agents+1, self.num_agents+6):
            unit_state = []
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                unit_state.append(x)
                unit_state.append(y)
                unit_state.append(np.clip((unit.heading%359)/359, 0, 1))
                unit_state.append(int(self.opponent_firing[i]))
            else:
                unit_state = np.zeros(4)
            
            state.extend(unit_state)

        state.insert(0, self.num_agents)
        state.insert(1, self.num_opps)
        state.insert(2, 1-self.steps/self.horizon) #t

        if len(state) < 93:
            state.extend(93-len(state))
        elif len(state) > 93:
            state = state[:93]

        return np.array(state, dtype=np.float32)

    def _all_distances(self, unit_id, with_af_id=False):
        def dist(agent_id, opp_id):
            return hypot(self.sim.get_unit(opp_id).position.lon - self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat - self.sim.get_unit(agent_id).position.lat)
        
        dists= []
        opps = [k for k in range(1, self.total_num+1)]
        opps.remove(unit_id)

        for i in opps:
            if self.sim.unit_exists(i):
                if with_af_id:
                    dists.append([i, dist(unit_id, i), np.clip(self._focus_angle(unit_id, i)/180, 0, 1)])
                else:
                    dists.append(dist(unit_id, i))
        if with_af_id:
            dists.sort(key=lambda x:x[1])
        else:  
            dists.sort()
        return dists

    def _closest_objects(self, agent_id):
        def dist(agent_id, opp_id):
            return hypot(self.sim.get_unit(opp_id).position.lon - self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat - self.sim.get_unit(agent_id).position.lat)
        
        def focus(agent_id, opp_id):
            return np.clip(self._focus_angle(agent_id, opp_id) / 180, 0, 1)

        order = []
        if agent_id <= self.num_agents:
            start = self.num_agents + 1
            end = self.total_num + 1
        else:
            start = 1
            end = self.num_agents + 1
        for i in range(start, end):
            if self.sim.unit_exists(i):
                order.append([i, dist(agent_id, i), focus(agent_id, i)])
            else:
                order.append([None, 100, 100])
        order.sort(key=lambda x:x[1])
        return order

    def _focus_angle(self, agent_id, opp_id):
        """
        Compute focus angle based on vector angles of current heading direction and position of the two airplanes. 
        """
        x = np.clip((np.dot(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ),sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]), np.array([self.sim.get_unit(opp_id).position.lon-self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat-self.sim.get_unit(agent_id).position.lat])))/(np.linalg.norm(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ),sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]))*np.linalg.norm(np.array([self.sim.get_unit(opp_id).position.lon-self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat-self.sim.get_unit(agent_id).position.lat]))+1e-10), -1, 1)
        return acos(x) * (180 / pi)

    def _hardcoded_opp(self, opp_id):
        """
        Deterministic opponents to fight against agents. They have slight randomness in heading and speed. 
        """
        d_agt = self._closest_objects(opp_id)
        opp_unit = self.sim.get_unit(opp_id)
        heading = opp_unit.heading
        fire = False
        speed = int(random.uniform(100, 400))
        if d_agt[0][0] is not None:
            sign = self._correct_angle_sign(opp_unit, self.sim.get_unit(d_agt[0][0]))
            r = random.uniform(0.7, 1.3)
            focus = self._focus_angle(opp_id, d_agt[0][0])
            if d_agt[0][1] > 0.008 and focus > 4:
                heading = (heading + r*sign*focus)%360
            if d_agt[0][1] > 0.05:
                speed = int(random.uniform(500, 800)) if focus < 30 else int(random.uniform(100, 500))
            fire = d_agt[0][1] < 0.03 and focus < 10
        return heading, speed, fire

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

    def _escaping_opps(self, unit):
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
        if self.opps_escaping_time >= 20:
            speed = int(random.uniform(100, 300))
        else:
            speed = int(random.uniform(400, 900))
        return heading, speed, bool(random.randint(0,1))

    def _sample_state(self, agent, i , r):
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

        return x, y, a

    def _reset_scenario(self):
        """
        create instances of airplane units (Rafale) and store them in dicts self.agents / self.opps.
        when instances get shot, they won't be removed / deleted, but just get invisible.
        """
        r = random.randint(1,2)
        for i in range(self.num_agents):
            x, y, a = self._sample_state("agent", i, r)

            agent_unit = Rafale(Position(y, x, 10_000), heading=a, speed=100)
            agent_unit.cannon_remain_secs = self.fire_max
            self.sim.add_unit(agent_unit)
            self.sim.record_unit_trace(agent_unit.id)
            self.alive_agents += 1

        for i in range(self.num_opps):
            x, y, a = self._sample_state("opp", i, r)

            opp_unit = Rafale(Position(y, x, 10_000), heading=a, speed=200)
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
                                    sum_angles(a.heading, Rafale.cannon_width_deg / 2.0),
                                    Rafale.cannon_range_km * 1000)
                d2 = geodetic_direct(a.position.lat, a.position.lon,
                                    sum_angles(a.heading, - Rafale.cannon_width_deg / 2.0),
                                    Rafale.cannon_range_km * 1000)
                objects.append(PolyLine([(a.position.lat, a.position.lon),
                                        (d1[0], d1[1]), (d2[0], d2[1]),
                                        (a.position.lat, a.position.lon)], line_width=1, dash=(1, 1),
                                        edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                        zorder=0))
        return objects
    