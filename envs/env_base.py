
import torch
import os
import random
import numpy as np
from pathlib import Path
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from math import sin, cos, acos, pi, hypot, radians, exp, sqrt
from warsim.scenplotter.scenario_plotter import PlotConfig, ColorRGBA, StatusMessage, TopLeftMessage, \
    Airplane, PolyLine, Drawable, Waypoint, Missile, ScenarioPlotter
from warsim.simulator.cmano_simulator import Position, CmanoSimulator, UnitDestroyedEvent
from warsim.simulator.ac1 import Rafale
from warsim.simulator.ac2 import RafaleLong
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

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

class HHMARLBaseEnv(MultiAgentEnv):
    """
    Base class for HHMARL 2D with core functionalities. 
    """
    def __init__(self, map_size):
        self._skip_env_checking = True
        self.steps = 0
        self.sim = None
        self.map_size = map_size
        self.map_limits = MapLimits(7.0, 5.0, 7.0+map_size, 5.0+map_size)

        self.alive_agents = 0
        self.alive_opps = 0
        self.rewards = {}

        # needed for combat behavior
        self.opp_to_attack = {}
        self.missile_wait = {}
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0

        # Plotting
        self.plt_cfg = PlotConfig()
        self.plt_cfg.units_scale = 20.0
        self.plotter = ScenarioPlotter(self.map_limits, dpi=200, config=self.plt_cfg)

        super().__init__()

    def reset(self, *, seed=None, options=None):
        """
        Reset scenario to a new random configuration.
        """
        self.steps = 0
        self.alive_agents = 0
        self.alive_opps = 0

        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.missile_wait = {i:0 for i in range(1, self.args.total_num+1)}
        self.opp_to_attack = {i:None for i in range(1, self.args.total_num+1)}

        self.sim = CmanoSimulator(num_units=self.args.num_agents, num_opp_units=self.args.num_opps)
        self._reset_scenario(options.get("mode", None))
        return
    
    def step(self, action):
        """
        Take one step for all agents in env.
        """
        self.rewards = {}
        if action:
            self._take_action(action)
        terminateds = truncateds = {}
        truncateds["__all__"] = terminateds["__all__"] = self.alive_agents <= 0 or self.alive_opps <= 0 or self.steps >= self.args.horizon
        return self.state(), self.rewards, terminateds, truncateds, {}
    
    def fight_state_values(self, agent_id, unit, opp, fri_id=None):
        """
        Fill the observation values for fight mode in low-level scene.
        opp = [opp_id, opp_dist]
        """
        state = []
        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.append(x)
        state.append(y)
        state.append(np.clip(unit.speed/unit.max_speed,0,1))
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        state.append(self._focus_angle(agent_id, opp[0],True))
        state.append(self._aspect_angle(opp[0], agent_id))
        state.append(self._heading_diff(agent_id, opp[0]))
        state.append(opp[1])
        state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
        if unit.ac_type == 1:
            state.append(np.clip(unit.missile_remain/unit.rocket_max, 0, 1))
            state.append(int(self.missile_wait[agent_id]==0))
            state.append(int(bool(unit.actual_missile)) or int(unit.cannon_current_burst_secs > 0))
        else:
            state.append(int(unit.cannon_current_burst_secs > 0))
        state.extend(self.opp_ac_values("fight", opp[0], agent_id, opp[1]))
        state.extend(self.friendly_ac_values(agent_id, fri_id))
        return state

    def esc_state_values(self, agent_id, unit, opps, fri_id=None):
        """
        Fill the observation values for escape mode in low-level scene.
        opps = [[opp_id1, opp_dist1], [opp_id2, opp_dist2], ...]
        """
        state = []
        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.append(x)
        state.append(y)
        state.append(np.clip(unit.speed/unit.max_speed,0,1))
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
        if unit.ac_type==1:
            state.append(np.clip(unit.missile_remain/unit.rocket_max, 0, 1))
        shot = unit.cannon_current_burst_secs > 0
        if unit.ac_type==1:
            shot = shot or bool(unit.actual_missile)
        state.append(int(shot))
        opp_state = []
        for opp in opps:
            opp_state.extend(self.opp_ac_values("esc", opp[0], agent_id, opp[1]))
            if len(opp_state) == 18:
                break
        if len(opp_state) < 18:
            opp_state.extend(np.zeros(18-len(opp_state)))
        state.extend(opp_state)
        state.extend(self.friendly_ac_values(agent_id, fri_id))
        return state

    def friendly_ac_values(self, agent_id, fri_id=None):
        """
        State of friendly aircraft w.r.t. agent or opp.
        """
        if not fri_id:
            return np.zeros(5)
        elif self.sim.unit_exists(fri_id):
            unit = self.sim.get_unit(fri_id)
            state = []
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            state.append(self._focus_angle(agent_id, fri_id, True))
            state.append(self._focus_angle(fri_id, agent_id, True))
            state.append(self._distance(agent_id, fri_id, True))
            return state
        else:
            return np.zeros(5)
        
    def opp_ac_values(self, mode, opp_id, agent_id, dist):
        """
        State of opponent aircraft w.r.t. agent or opp.
        """
        state = []
        unit = self.sim.get_unit(opp_id)
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.append(x)
        state.append(y)
        state.append(np.clip(unit.speed/unit.max_speed,0,1))
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        state.append(self._heading_diff(opp_id, agent_id))
        if mode == "fight":
            state.append(self._focus_angle(opp_id, agent_id,True))
            state.append(self._aspect_angle(agent_id, opp_id))
        else:
            state.append(self._focus_angle(agent_id, opp_id, True))
            state.append(self._focus_angle(opp_id, agent_id, True))
        if mode == "HighLevel":
            state.append(self._aspect_angle(agent_id, opp_id))
            state.append(self._aspect_angle(opp_id, agent_id))
        state.append(dist)
        if mode != "HighLevel":
            shot = unit.cannon_current_burst_secs > 0
            if unit.ac_type==1:
                shot = shot or bool(unit.actual_missile)
            state.append(int(shot))
        return state

    def _take_base_action(self, mode, unit, unit_id, opp_id, actions, rewards=None):
        """
        Take basic combat step in env (and assign some rewards for escape).
        """
        unit.set_heading((unit.heading + (actions[unit_id][0]-6)*15)%360) #relative heading
        unit.set_speed(100+((unit.max_speed-100)/8)*actions[unit_id][1]) #absolute speed

        if bool(actions[unit_id][2]) and unit.cannon_remain_secs > 0:
            unit.fire_cannon()
            if mode=="LowLevel" and unit_id <= self.args.num_agents:
                if self.agent_mode == "escape" and unit.cannon_remain_secs < 90:
                    rewards[unit_id] -= 0.1

        if unit.ac_type == 1 and bool(actions[unit_id][3]):
            if opp_id and unit.missile_remain > 0 and not unit.actual_missile and self.missile_wait[unit_id] == 0:
                unit.fire_missile(unit, self.sim.get_unit(opp_id), self.sim)
                self.missile_wait[unit_id] = random.randint(7,17) if mode == "LowLevel" else random.randint(8, 12)
                if mode=="LowLevel" and unit_id <= self.args.num_agents:
                    if self.agent_mode == "escape" and unit.missile_remain < 3:
                        rewards[unit_id] -= 0.1

        if self.missile_wait[unit_id] > 0 and not bool(unit.actual_missile):
            self.missile_wait[unit_id] = self.missile_wait[unit_id] -1

        return rewards

    def _combat_rewards(self, events, opp_stats=None, mode="LowLevel"):
        """"
        Calculating Rewards. 
        First check for out-of-boundary, then killing rewards.
        """
        rews = {a:[] for a in range(1,self.args.num_agents+1)}
        destroyed_ids = []
        s=self.args.rew_scale
        kill_event = False

        #out-of-boundary punishment
        for i in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if not self.map_limits.in_boundary(u.position.lat, u.position.lon):
                    self.sim.remove_unit(i)
                    kill_event = True
                    if i <= self.args.num_agents:
                        p = -5 if mode=="LowLevel" else -2
                        rews[i].append(p*s)
                        destroyed_ids.append(i)
                        self.alive_agents -= 1
                    else:
                        self.alive_opps -= 1

        #event rewards
        for ev in events:

            # agent kill
            if ev.unit_killer.id <= self.args.num_agents:

                #killed opp
                if ev.unit_destroyed.id in range(self.args.num_agents+1, self.args.total_num+1):
                    if mode=="LowLevel":
                        if self.agent_mode == "fight":
                            if ev.origin.id >= self.args.total_num+1: #killed by rocket
                                rews[ev.unit_killer.id].append(self._shifted_range(ev.unit_killer.missile_remain/ev.unit_killer.rocket_max, 0,1, 1,1.5)*s)
                            else:
                                rews[ev.unit_killer.id].append((self._shifted_range(ev.unit_killer.cannon_remain_secs/ev.unit_killer.cannon_max, 0,1, 0.5,1) + self._shifted_range(opp_stats[ev.unit_killer.id][0], 0,1, 0.5,1))*s)
                        else:
                            # no reward for killing in escape mode
                            pass
                            #rews[ev.unit_killer.id].append(1)
                    else:
                        #constant reward for killing in High-Level Env
                        rews[ev.unit_killer.id].append(1)

                    self.alive_opps -= 1

                #friendly kill
                elif ev.unit_destroyed.id <= self.args.num_agents:
                    if mode=="LowLevel":
                        rews[ev.unit_killer.id].append(-2*s)
                        if self.args.friendly_punish:
                            rews[ev.unit_destroyed.id].append(-2*s)
                            destroyed_ids.append(ev.unit_destroyed.id)
                    self.alive_agents -= 1

            # opp kill
            elif ev.unit_killer.id in range(self.args.num_agents+1, self.args.total_num+1):
                if ev.unit_destroyed.id <= self.args.num_agents:
                    p = -2 if mode=="LowLevel" else -1
                    rews[ev.unit_destroyed.id].append(p*s)
                    destroyed_ids.append(ev.unit_destroyed.id)
                    self.alive_agents -= 1
                elif ev.unit_destroyed.id in range(self.args.num_agents+1, self.args.total_num+1):
                    self.alive_opps -= 1

            kill_event = True

        return rews, destroyed_ids, kill_event

    def _get_policies(self, mode):
        """
        Restore torch policies for fictitious self-play.
        """
        policy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'policies')
        self.policy = {}
        if mode == "LowLevel":
            if self.args.level == 4:
                self.policy = {"fight_1": torch.load(os.path.join(policy_dir, 'L3_AC1.pt')), "fight_2": torch.load(os.path.join(policy_dir, 'L3_AC2.pt'))}
            else:
                self.policies = {}
                for i in range(3,6):
                    if i <= 4:
                        self.policies[i] = {"fight_1": torch.load(os.path.join(policy_dir, f'L{i}_AC1.pt')), "fight_2": torch.load(os.path.join(policy_dir, f'L{i}_AC2.pt'))}
                    else:
                        self.policies[i] = {"escape_1": torch.load(os.path.join(policy_dir, 'Esc_AC1.pt')), "escape_2": torch.load(os.path.join(policy_dir, 'Esc_AC2.pt'))}
        else:
            self.policy["fight_1"] = torch.load(os.path.join(policy_dir, 'L5_AC1.pt'))
            self.policy["fight_2"] = torch.load(os.path.join(policy_dir, 'L5_AC2.pt'))
            self.policy["escape_1"] = torch.load(os.path.join(policy_dir, 'Esc_AC1.pt'))
            self.policy["escape_2"] = torch.load(os.path.join(policy_dir, 'Esc_AC2.pt'))
        return

    def _policy_actions(self, policy_type, agent_id, unit):
        """
        Apply self-play actions from previous, learned policy.
        We call method get_torch_action() to compute actions manually, 
        because there is an inconsistent behaviour when calling Policy.compute_single_action() from Ray (version 2.4)
        """
        actions = {}

        def obs_tens(obs):
            if unit.ac_type == 1:
                return {
                    "obs_1_own": torch.tensor(obs),
                    "obs_2": torch.zeros((1,OBS_AC2 if policy_type=="fight" else OBS_ESC_AC2)),
                    "act_1_own": torch.zeros((1,ACTION_DIM_AC1)),
                    "act_2": torch.zeros((1,ACTION_DIM_AC2)),
                }
            else:
                return {
                    "obs_1_own": torch.tensor(obs),
                    "obs_2": torch.zeros((1,OBS_AC1 if policy_type=="fight" else OBS_ESC_AC1)),
                    "act_1_own": torch.zeros((1,ACTION_DIM_AC2)),
                    "act_2": torch.zeros((1,ACTION_DIM_AC1)),
                }

        def get_torch_action(inp):
            """
            Manual computation of action based on Categorical distribution.
            """
            in_lens = np.array([13,9,2,2]) if unit.ac_type == 1 else np.array([13,9,2])
            inputs_split = inp.split(tuple(in_lens), dim=1)
            cats = [torch.distributions.categorical.Categorical(logits=input_) for input_ in inputs_split]
            arr = [torch.argmax(cat.probs, -1) for cat in cats]
            action = torch.stack(arr, dim=1)
            return np.array(action[0])

        state = self.lowlevel_state(policy_type, agent_id, unit=unit)

        with torch.no_grad():
            out = self.policy[f"{policy_type}_{unit.ac_type}"](
                input_dict = {"obs": obs_tens(np.expand_dims(state[agent_id], axis=0))},
                state=[torch.tensor(0)],
                seq_lens=torch.tensor([1])
            )
        actions[agent_id] = get_torch_action(out[0])
        return actions

    def _nearby_object(self, agent_id, friendly=False):
        """
        Return a sorted list with id's and distances to opponents/friendly aircraft. 
        """
        order = []
        if friendly:
            f = list(range(1,self.args.num_agents+1)) if agent_id <= self.args.num_agents else list(range(self.args.num_agents+1, self.args.total_num+1))
            f.remove(agent_id)
            for i in f:
                if self.sim.unit_exists(i):
                    order.append([i, self._distance(agent_id, i, True)])
        else:
            if agent_id <= self.args.num_agents:
                start = self.args.num_agents + 1
                end = self.args.total_num + 1
            else:
                start = 1
                end = self.args.num_agents + 1
            for i in range(start, end):
                if self.sim.unit_exists(i):
                    order.append([i, self._distance(agent_id, i, True), self._distance(agent_id, i)])
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
        return self._shifted_range(d, 0, sqrt(2*self.map_size**2), 0, 1) if norm else d

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

    def _sample_state(self, agent, i, r):
        x = 0
        y = 0
        a = 0
        if agent == "agent":
            if self.args.level == 1:
                if r == 1:
                    x = random.uniform(7.12, 7.14)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                    a = random.randint(30, 150)
                elif r == 2:
                    x = random.uniform(7.16, 7.17)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                    a = random.randint(200, 330)
            elif self.args.level == 2:
                if r == 1:
                    x = random.uniform(7.08, 7.13)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 180)
                elif r == 2:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(180, 359)
            elif self.args.level >= 3:
                if r == 1:
                    x = random.uniform(7.07, 7.12)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(0, 270)
                elif r == 2:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(90, 359)

        elif agent == "opp":
            if self.args.level == 1:
                if r == 1:
                    x = random.uniform(7.16, 7.17)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                elif r == 2:
                    x = random.uniform(7.12, 7.14)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
            elif self.args.level == 2:
                if r == 1:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 359)
                elif r == 2:
                    x = random.uniform(7.08, 7.13)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 359)
            elif self.args.level >= 3:
                if r == 1:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(0, 359)
                elif r == 2:
                    x = random.uniform(7.07, 7.12)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(0, 359)
        
        return x,y,a

    def _reset_scenario(self, mode):
        """
        create aircraft units (Rafale and RafaleLong).
        """
        r = random.randint(1,2) #chose sides
        for group, count in [("agent", self.args.num_agents), ("opp", self.args.num_opps)]:
            for i in range(count):
                x, y, a = self._sample_state(group, i, r)
                #at least one aircraft type (ac) per group
                ac = i+1 if i <=1 else random.randint(1,2)
                if ac == 1:
                    unit = Rafale(Position(y, x, 10_000), heading=a, speed=0 if self.args.level<=2 and group=="opp" else 100, group="agent", friendly_check=self.args.friendly_kill)
                else:
                    unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=0 if self.args.level<=2 and group=="opp" else 100, group="agent", friendly_check=self.args.friendly_kill)

                if mode == "LowLevel":
                    if self.args.level <= 4 and group == "opp":
                        unit.missile_remain = unit.rocket_max = 8
                        unit.cannon_max = unit.cannon_remain_secs = 400
                    elif self.args.level == 5:
                        unit.missile_remain = unit.rocket_max = 6
                        unit.cannon_max = unit.cannon_remain_secs = 300
                else:
                    unit.cannon_max = unit.cannon_remain_secs = 300
                    if ac == 1:
                        unit.missile_remain = unit.rocket_max = 8

                self.sim.add_unit(unit)
                self.sim.record_unit_trace(unit.id)
                if group == "agent":
                    self.alive_agents += 1
                else:
                    self.alive_opps += 1

    def _plot_airplane(self, a: Rafale, side: str, path=True, use_backup=False, u_id=0):
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

    def plot(self, out_file: Path, paths=True):
        """
        Draw current scenario.
        """
        objects = [
            StatusMessage(self.sim.status_text),
            TopLeftMessage(self.sim.utc_time.strftime("%Y %b %d %H:%M:%S"))
        ]
        for i in range(1, self.args.num_agents+self.args.num_opps+1):
            col = 'blue' if i<=self.args.num_agents else 'red'
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                objects.extend(self._plot_airplane(unit, col, paths))
            else:
                objects.extend(self._plot_airplane(None, col, paths, True, i))
        for i in range(self.args.total_num+1, self.args.total_num*5+2):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                col = "blue" if unit.source.id <= self.args.num_agents else "red"
                objects.append(
                    Missile(unit.position.lat, unit.position.lon, unit.heading, edge_color=colors[f'{col}_outline'], fill_color=colors[f'{col}_fill'],
                    info_text=f"m_{i}", zorder=0),
                )
        self.plotter.to_png(str(out_file), objects)

