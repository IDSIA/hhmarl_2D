"""
Low-Level Environment for HHMARL 2D Aircombat.
"""
import random
import numpy as np
from gymnasium import spaces
from .env_base import HHMARLBaseEnv

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

class LowLevelEnv(HHMARLBaseEnv):
    """
    Low-Level Environment for Aircombat Maneuvering.
    """
    def __init__(self, env_config):
        self.args = env_config.get("args", None)
        self.agent_mode = self.args.agent_mode
        self.opp_mode = self.args.opp_mode

        if self.agent_mode == "fight":
            self.obs_dim_map = {1: OBS_AC1, 2:OBS_AC2, 3:OBS_AC1, 4:OBS_AC2}
        else:
            # escape state-space
            self.obs_dim_map = {1: OBS_ESC_AC1, 2:OBS_ESC_AC2, 3:OBS_ESC_AC1, 4:OBS_ESC_AC2}

        self._obs_space_in_preferred_format = True
        self.observation_space = spaces.Dict({
            1: spaces.Box(low=np.zeros(self.obs_dim_map[1]), high=np.ones(self.obs_dim_map[1]), dtype=np.float32),
            2: spaces.Box(low=np.zeros(self.obs_dim_map[2]), high=np.ones(self.obs_dim_map[2]), dtype=np.float32),
            3: spaces.Box(low=np.zeros(self.obs_dim_map[3]), high=np.ones(self.obs_dim_map[3]), dtype=np.float32),
            4: spaces.Box(low=np.zeros(self.obs_dim_map[4]), high=np.ones(self.obs_dim_map[4]), dtype=np.float32)
            }
        )
        self._action_space_in_preferred_format = True
        self.action_space = spaces.Dict({
            1: spaces.MultiDiscrete([13,9,2,2]), 
            2: spaces.MultiDiscrete([13,9,2]), 
            3: spaces.MultiDiscrete([13,9,2,2]), 
            4: spaces.MultiDiscrete([13,9,2])
            }
        )
        self._agent_ids = set(range(1,self.args.total_num+1))

        super().__init__(self.args.map_size)

        # fictitious Self-Play (sp), starting from L4 on
        if self.args.level >= 4:
           self._get_policies("LowLevel")

    def reset(self, *, seed=None, options=None):
        super().reset(options={"mode":"LowLevel"})
        if self.args.level == 5:
            #randomly select opponent behavior every new episode in L5.
            k = random.randint(3,5)
            self.policy = self.policies[k]
            self.opp_mode = "escape" if k == 5 else "fight"
        return self.state(), {}
    
    def state(self):
        return self.lowlevel_state(self.agent_mode)

    def lowlevel_state(self, mode, agent_id=None, **kwargs):
        """
        Current observation in fight / esc mode, stored in state_dict. 
        Destroyed agent's observation is filled with zeros, needed for Ray callback (centralized critic).
        """

        def fri_ac_id(agent_id):
            if agent_id<=self.args.num_agents:
                return 1 if agent_id == 2 else 2
            else:
                return 3 if agent_id == 4 else 4

        #need to define obs_dim again, because in L5, agent is in fight mode and opp may switch to esc mode.
        obs_dim = {1: OBS_ESC_AC1, 2: OBS_ESC_AC2, 3: OBS_ESC_AC1, 4: OBS_ESC_AC2} if mode == "escape" else self.obs_dim_map
        state_dict = {}

        if agent_id:
            start = agent_id
            end = agent_id +1
        else:
            start = 1
            end = self.args.num_agents+1

        for ag_id in range(start, end):
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                opps = self._nearby_object(ag_id)
                if opps:
                    unit = self.sim.get_unit(ag_id)
                    state = self.fight_state_values(ag_id, unit, opps[0], fri_ac_id(ag_id)) if mode == "fight" else self.esc_state_values(ag_id, unit, opps, fri_ac_id(ag_id))
                    self.opp_to_attack[ag_id] = opps[0][0]
                    assert len(state) == obs_dim[ag_id], f"{mode} state len {len(state)} is not as required ({obs_dim[ag_id]}) for agent {ag_id}"
                else:
                    state = np.zeros(obs_dim[ag_id], dtype=np.float32)
            else:
                state = np.zeros(obs_dim[ag_id], dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _take_action(self, action):
        """
        Apply actions to agents and opponents and get rewards.
        Opponent behavior: 
            -L1 and L2 random
            -L3 engage nearby agent (method _hardcoded_opp())
            -L4 previous policy (L3)
            -L5 previous policies + escape.
        """
        self.steps += 1
        rewards = {}
        opp_stats = {}

        def __opp_level1(unit, unit_id):
            if not unit.actual_missile and self.steps % 40 in range(3) and bool(random.randint(0,1)) and self.missile_wait[i] == 0 and unit.ac_type == 1:
                d_ag = self._nearby_object(unit_id)
                if d_ag:
                    unit.fire_missile(unit, self.sim.get_unit(d_ag[0][0]), self.sim)
                    self.missile_wait[unit_id] = 5

        def __opp_level2(unit, unit_id):
            unit.fire_cannon()
            if self.steps<=5 or self.steps % random.randint(35,45) <= 5:
                r = random.randint(0,1)
                unit.set_heading((unit.heading + ((-1)**r)*90)%360) #+90 = max turn right, -90 = max turn left
                s = 100 + random.randint(0,4)*75
                unit.set_speed(s)
            if not unit.actual_missile and self.steps % 40 in range(3) and bool(random.randint(0,1)) and self.missile_wait[unit_id] == 0 and unit.ac_type == 1:
                d_ag = self._nearby_object(unit_id)
                if d_ag:
                    unit.fire_missile(unit, self.sim.get_unit(d_ag[0][0]), self.sim)
                    self.missile_wait[unit_id] = 5

        def __opp_level3(unit, unit_id):
            if self.steps % 60 == 0 and not self.hardcoded_opps_escaping:
                self.hardcoded_opps_escaping = bool(random.randint(0, 1))
                if self.hardcoded_opps_escaping:
                    self.opps_escaping_time = int(random.uniform(20, 30))

            if self.hardcoded_opps_escaping:
                #in order to prevent from keeping circulating to tail of agent, we randomly set opponents to escape.
                opp, heading, speed, fire, fire_missile, _ = self._escaping_opp(unit)
                self.opps_escaping_time -= 1
                if self.opps_escaping_time <= 0:
                    self.hardcoded_opps_escaping = False
            else:
                opp, heading, speed, fire, fire_missile, _ = self._hardcoded_opp(unit, unit_id)
            unit.set_heading(heading)
            unit.set_speed(speed)
            if fire:
                unit.fire_cannon()
            if fire_missile and opp and not unit.actual_missile and self.missile_wait[unit_id] == 0 and unit.ac_type == 1:
                unit.fire_missile(unit, self.sim.get_unit(opp), self.sim)
                self.missile_wait[unit_id] = 10

        for i in range(1, self.args.total_num+1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if i <= self.args.num_agents or self.args.level >= 4:
                    if i >= self.args.num_agents+1:
                        actions = self._policy_actions(policy_type=self.opp_mode, agent_id=i, unit=u)
                    else:
                        actions = action
                        rewards[i] = 0
                        if self.sim.unit_exists(self.opp_to_attack[i]):
                            opp_stats[i] = [self._focus_angle(self.opp_to_attack[i], i, True), self._distance(i, self.opp_to_attack[i])]

                    rewards = self._take_base_action("LowLevel",u,i,self.opp_to_attack[i],actions,rewards)

                else:
                    if self.args.level == 1:
                        __opp_level1(u, i)

                    elif self.args.level == 2:
                        __opp_level2(u, i)

                    elif self.args.level == 3:
                        __opp_level3(u, i)

        # self.sim.do_tick simulates the dynamics
        self.rewards = self._get_rewards(rewards, self.sim.do_tick(), opp_stats)
        return

    def _get_rewards(self, rewards, events, opp_stats):
        """
        Calculating Rewards. 
        First check for out-of-boundary, then killing rewards.
        rewards are collected in dict 'rews' in order to sum them together and 
        possibly add a global fraction 'glob_frac' (to incorporate rewads of cooperating agents).

        It's also possible to scale rewards by adjusting 'rew_scale' in config.
        """
        rews, destroyed_ids, _ = self._combat_rewards(events, opp_stats)

        if self.agent_mode == "escape":
            for i in range(1, self.args.num_agents+1):
                if self.sim.unit_exists(i):
                    u = self.sim.get_unit(i)
                    x = u.position.lon
                    y = u.position.lat
                    #speed-bound
                    if x > 7.025 and x < 7.275 and y > 5.025 and y < 5.275:
                        if u.ac_type == 1 and u.speed > 600:
                            rews[i].append(0.02)
                        if u.ac_type == 2 and u.speed > 450:
                            rews[i].append(0.02)
                    else:
                        rews[i].append(-0.02)
                    for j in range(self.args.num_agents+1, self.args.total_num+1):
                        if self.sim.unit_exists(j):
                            #u = self.sim.get_unit(i)
                            d = self._distance(i, j)
                            if d < 0.06:
                                rews[i].append(-0.02)
                                if self._focus_angle(j, i) < 15: #25
                                    rews[i].append(-0.05)
                            elif d > 0.13:
                                rews[i].append(0.02)
                            # speed-dist
                            # if d > 0.1:
                            #     if u.ac_type == 1 and u.speed > 600: #550
                            #         rews[i].append(0.02)
                            #     elif u.ac_type == 2 and u.speed > 500: #400
                            #         rews[i].append(0.02)
                
        #sum all collected rewards together
        for i in range(1, self.args.num_agents+1):
            if self.sim.unit_exists(i) or i in destroyed_ids:
                if self.args.glob_frac > 0 and self.agent_mode == "fight":
                    rewards[i] += sum(rews[i]) + self.args.glob_frac*sum(rews[i%2+1])
                else:
                    rewards[i] += sum(rews[i])

        return rewards
    
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

    def _hardcoded_opp(self, opp_unit, opp_id):
        """
        Deterministic opponents to fight against agents. They have slight randomness in heading and speed. 
        """
        d_agt = self._nearby_object(opp_id)
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