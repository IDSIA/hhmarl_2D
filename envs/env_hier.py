"""
High-Level Environment for HHMARL 2D Aircombat.
Low Level agent policies are included in this env.
"""

import os
import random
import numpy as np
from fractions import Fraction
from gymnasium import spaces
from .env_base import HHMARLBaseEnv

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

N_OPP_HL = 2 #change for sensing
OBS_OPP_HL = 10
OPP_SIZE = N_OPP_HL*OBS_OPP_HL
OBS_FRI_HL = 5
FRI_SIZE = 2 * OBS_FRI_HL
OBS_HL = 14 + N_OPP_HL*OBS_OPP_HL

class HighLevelEnv(HHMARLBaseEnv):
    """
    High-Level Environment for Aircombat Maneuvering.
    """
    def __init__(self, env_config):
        self.args = env_config.get("args", None)
        self.n_sub_steps = 15
        self.min_sub_steps = 10

        self.observation_space = spaces.Box(low=np.zeros(OBS_HL), high=np.ones(OBS_HL), dtype=np.float32)
        self.action_space = spaces.Discrete(N_OPP_HL+1)
        self._agent_ids = set(range(1,self.args.num_agents+1))
        self.commander_actions = None

        super().__init__(self.args.map_size)
        self._get_policies("HighLevel")

    def reset(self, *, seed=None, options=None):
        super().reset(options={"mode":"HighLevel"})
        self.commander_actions = None
        return self.state(), {}

    def state(self):
        """
        High Level state for commander.
        self.opp_to_attack[agent_id] = [[opp_id1, opp_dist2], [opp_id2, opp_dist2], ...]
        """
        state_dict = {}

        for ag_id in range(1, self.args.total_num+1):
            self.opp_to_attack[ag_id] = []
            if ag_id <= self.args.num_agents:
                if self.sim.unit_exists(ag_id):
                    state = []
                    opps = self._nearby_object(ag_id)
                    if opps:
                        unit = self.sim.get_unit(ag_id)
                        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                        state.append(x)
                        state.append(y)
                        state.append(np.clip(unit.speed/unit.max_speed,0,1))
                        state.append(np.clip((unit.heading%359)/359, 0, 1))
                        opp_state = []
                        for opp in opps:
                            opp_state.extend(self.opp_ac_values("HighLevel", opp[0], ag_id, opp[1]))
                            self.opp_to_attack[ag_id].append(opp) #store opponents, Commander decides which to attack
                            if len(opp_state) == OPP_SIZE:
                                break
                        if len(opp_state) < OPP_SIZE:
                            opp_state.extend(np.zeros(OPP_SIZE-len(opp_state)))
                        fri_state = []
                        fri = self._nearby_object(ag_id, True)
                        for f in fri:
                            fri_state.extend(self.friendly_ac_values(ag_id, f[0]))
                            if len(fri_state) == FRI_SIZE:
                                break
                        if len(fri_state) < FRI_SIZE:
                            fri_state.extend(np.zeros(FRI_SIZE-len(fri_state)))
                        state.extend(opp_state)
                        state.extend(fri_state)
                        assert len(state) == OBS_HL, f"Hierarchy state len {len(state)} is not as required ({OBS_HL}) for agent {ag_id}"
                    else:
                        state = np.zeros(OBS_HL, dtype=np.float32)
                else:
                    state = np.zeros(OBS_HL, dtype=np.float32)
                state_dict[ag_id] = np.array(state, dtype=np.float32)
            else:
                if self.sim.unit_exists(ag_id):
                    # for opponents, only need to store the closest agent
                    # explicitly exclude from the loop above to not fill values in state_dict
                    self.opp_to_attack[ag_id] = self._nearby_object(ag_id)
        return state_dict

    def lowlevel_state(self, mode, agent_id, unit):
        """
        Lowlevel state for fight or escape.
        the opponent to attack is fixed for the whole low-level cycle, 
        but the closest friendly aircraft will be determined new every round.
        """
        fri_id = self._nearby_object(agent_id, friendly=True)
        fri_id = fri_id[0][0] if fri_id else None
        if mode == "fight":
            state = self.fight_state_values(agent_id, unit, self.opp_to_attack[agent_id][self.commander_actions[agent_id]-1], fri_id)
        else:
            state = self.esc_state_values(agent_id, unit, self.opp_to_attack[agent_id], fri_id)
        return {agent_id:np.array(state, dtype=np.float32)}

    def _take_action(self, commander_actions):
        s = 0
        kill_event = False
        situation_event = False
        rewards = {}
        self.commander_actions = commander_actions
        self.min_sub_steps = 10 #random.randint(10,16)

        # Select opp to attack and assess action
        rewards = self._action_assess(rewards)

        while s <= self.n_sub_steps and not kill_event and not situation_event:
            for i in range(1, self.args.total_num+1):
                if self.sim.unit_exists(i):
                    u = self.sim.get_unit(i)
                    actions = self._policy_actions(policy_type="escape" if self.commander_actions[i]==0 else "fight", agent_id=i, unit=u)
                    self._take_base_action("HighLevel", u, i, self.opp_to_attack[i][self.commander_actions[i]-1][0], actions)

            rewards, kill_event = self._get_rewards(rewards, self.sim.do_tick())
            if s > self.min_sub_steps:
                #take at least min_sub_steps until checking for surrounding event
                situation_event = self._surrounding_event()

            s += 1
            self.steps += 1
        self.rewards = rewards
        return

    def _action_assess(self, rewards):
        """
        Select the opponent to attack based on commander action.
        Punish for chosing not existing opponent, reward for choosing to fight in favourable situation.
        If an opponent is chosen that does not exist (observation filled with zeros), 
        we assign the closest opponent to attack -> self.commander_actions[i] = 1 (because 0 is escape)

        self.commander_actions will be expanded to include closest agents to attack w.r.t opponents
        """
        for i in range(1,self.args.total_num+1):
            if self.sim.unit_exists(i):
                if i <= self.args.num_agents:
                    rewards[i] = 0
                    if self.commander_actions[i] > 0:
                        try:
                            opp_id = self.opp_to_attack[i][self.commander_actions[i]-1][0]
                        except:
                            opp_id = None
                            self.commander_actions[i] = 1

                        if not opp_id: rewards[i] = -0.1
                        if self.args.hier_action_assess and opp_id:
                            if self._distance(i, opp_id) < 0.1 and self._focus_angle(i, opp_id) < 15 and self._focus_angle(opp_id, i) > 40:
                                rewards[i] = 0.1
                            else:
                                rewards[i] = 0
                    else:
                        if self.args.hier_action_assess:
                            cl_opp = self.opp_to_attack[i][0][0]
                            if self._distance(cl_opp, i) < 0.1 and self._focus_angle(cl_opp, i) < 15 and self._focus_angle(i, cl_opp) > 40:
                                rewards[i] = 0.1
                else:
                    # determine if opponents select pi_fight with fight probability
                    p_of = Fraction(self.args.hier_opp_fight_ratio, 100).limit_denominator().as_integer_ratio()
                    fight = bool(random.choices([0, 1], weights=[p_of[1]-p_of[0], p_of[0]], k=1)[0])
                    if fight:
                        possible_agent_ids = len(self.opp_to_attack[i])
                        if possible_agent_ids > 1 and bool(random.choices([0, 1], weights=[1, 3], k=1)[0]):
                            # randomly select another agent to attack
                            ag_id = random.randint(2,possible_agent_ids)
                        else:
                            ag_id = 1
                    else: ag_id = 0 #escape
                    # define commander actions for opponents
                    self.commander_actions[i] = ag_id
            else:
                if i <= self.args.num_agents: rewards[i] = 0
                self.commander_actions[i] = None
        return rewards

    def _surrounding_event(self):
        def eval_event(ag_id, opp_id):
            if self._distance(ag_id, opp_id) < 0.1:
                if self._focus_angle(ag_id, opp_id) < 15 or self._focus_angle(opp_id, ag_id) < 15:
                    return True
            return False
        
        event = False
        for i in range(1, self.args.num_agents+1):
            for j in range(self.args.num_agents+1, self.args.total_num+1):
                if self.sim.unit_exists(i) and self.sim.unit_exists(j):
                    event = eval_event(i, j)
                if event:
                    break
            if event:
                break
        return event

    def _get_rewards(self, rewards, events):
        rews, destroyed_ids, kill_event = self._combat_rewards(events, mode="HighLevel")

        #sum all collected rewards together
        for i in range(1, self.args.num_agents+1):
            if self.sim.unit_exists(i) or i in destroyed_ids:
                if self.args.glob_frac > 0:
                    #incorporate rewards of other aircraft
                    ids = list(range(1,self.args.num_agents+1))
                    ids.remove(i)
                    rewards[i] += sum(rews[i]) + self.args.glob_frac*sum(sum(rews[j]) for j in ids)
                else:
                    rewards[i] += sum(rews[i])

        return rewards, kill_event

    def _sample_state(self, agent, i, r):
        x = 0
        y = 0
        a = 0
        if agent == "agent":
            if r == 1:
                x = random.uniform(7.07, 7.22)
                y = random.uniform(5.07 + i*(0.4/self.args.num_agents), 5.12 + i*(0.4/self.args.num_agents))
                a = random.randint(0, 359)
            elif r == 2:
                x = random.uniform(7.28, 7.43)
                y = random.uniform(5.07 + i*(0.4/self.args.num_agents), 5.12 + i*(0.4/self.args.num_agents))
                a = random.randint(0, 359)

        else:
            if r == 1:
                x = random.uniform(7.28, 7.43)
                y = random.uniform(5.07 + i*(0.4/self.args.num_opps), 5.12 + i*(0.4/self.args.num_opps))
                a = random.randint(0, 359)
            elif r == 2:
                x = random.uniform(7.07, 7.22)
                y = random.uniform(5.07 + i*(0.4/self.args.num_opps), 5.12 + i*(0.4/self.args.num_opps))
                a = random.randint(0, 359)
        
        return x,y,a
