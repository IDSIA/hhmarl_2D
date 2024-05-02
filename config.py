import argparse
import os
import datetime

class Config(object):
    """
    Configurations for HHMARL Training. 
    Mode 0 = Low-level training
    Mode 1 = High-level training
    Mode 2 = Evaluation
    """
    def __init__(self, mode:int):
        self.mode = mode
        parser = argparse.ArgumentParser(description='HHMARL2D Training Config')

        # training mode
        parser.add_argument('--level', type=int, default=1, help='Training Level')
        parser.add_argument('--horizon', type=int, default=500, help='Length of horizon')
        parser.add_argument('--agent_mode', type=str, default="fight", help='Agent mode: Fight or Escape')
        parser.add_argument('--num_agents', type=int, default=2 if mode==0 else 4, help='Number of (trainable) agents')
        parser.add_argument('--num_opps', type=int, default=2 if mode==0 else 4, help='Number of opponents')
        parser.add_argument('--total_num', type=int, default=4 if mode==0 else 8, help='Total number of aircraft')
        parser.add_argument('--hier_opp_fight_ratio', type=int, default=75, help='Opponent fight policy selection probability [in %].')

        # env & training params
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')
        parser.add_argument('--restore', type=bool, default=False, help='Restore from model')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')
        parser.add_argument('--log_name', type=str, default=None, help='Experiment Name, defaults to Commander + date & time.')
        parser.add_argument('--log_path', type=str, default=None, help='Full Path to actual trained model')

        parser.add_argument('--gpu', type=float, default=0)
        parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel samplers')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=2000 if mode==0 else 1000, help='PPO train batch size')
        parser.add_argument('--mini_batch_size', type=int, default=256, help='PPO train mini batch size')
        parser.add_argument('--map_size', type=float, default=0.3 if mode==0 else 0.5, help='Map size -> *100 = [km]')

        # rewards
        parser.add_argument('--glob_frac', type=float, default=0, help='Fraction of reward sharing')
        parser.add_argument('--rew_scale', type=int, default=1, help='Reward scale')
        parser.add_argument('--esc_dist_rew', type=bool, default=False, help='Activate per-time-step reward for Escape Training.')
        parser.add_argument('--hier_action_assess', type=bool, default=True, help='Give action rewards to guide hierarchical training.')
        parser.add_argument('--friendly_kill', type=bool, default=True, help='Consider friendly kill or not.')
        parser.add_argument('--friendly_punish', type=bool, default=False, help='If friendly kill occurred, if both agents to punish.')

        # eval
        parser.add_argument('--eval_info', type=bool, default=True if mode==2 else False, help='Provide eval statistic in step() function or not. Dont change for evaluation.')
        parser.add_argument('--eval_hl', type=bool, default=False, help='True=evaluation with Commander, False=evaluation of low-level policies.')
        parser.add_argument('--eval_level_ag', type=int, default=5, help='Agent low-level for evaluation.')
        parser.add_argument('--eval_level_opp', type=int, default=4, help='Opponent low-level for evaluation.')
        
        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        self.args = parser.parse_args()
        self.set_metrics()

    def set_metrics(self):

        #self.args.log_name = f'Commander_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'
        self.args.log_name = f'L{self.args.level}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}' if self.mode == 0 else f'Commander_{self.args.num_agents}_vs_{self.args.num_opps}'
        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if not self.args.restore and self.mode==0:
            if self.args.agent_mode == "fight" and os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L{self.args.level-1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}')):
                self.args.restore = True
            elif self.args.agent_mode == "escape" and os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L3_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}')):
                self.args.restore = True

        if self.args.restore:
            if self.args.restore_path is None:
                if self.mode == 0:
                    try:
                        if self.args.agent_mode=="fight":
                            # take previous pi_fight
                            self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', f'L{self.args.level-1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}', 'checkpoint')
                        else:
                            # escape-vs-pi_fight
                            self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', f'L3_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}', 'checkpoint')
                    except:
                        raise NameError(f'Could not restore previous {self.args.agent_mode} policy. Check restore_path.')
                else:
                    raise NameError('Specify full restore path to Commander Policy.')

        if self.args.agent_mode == "escape" and self.mode==0:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L3_escape_2-vs-2')):
                self.args.level = 3
            else:
                self.args.level = 5
            self.args.log_name = f'L{self.args.level}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}'
            self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if self.mode == 0:
            horizon_level = {1: 150, 2:200, 3:300, 4:350, 5:400}
            self.args.horizon = horizon_level[self.args.level]
        else:
            self.args.horizon = 500

        if self.mode == 2 and self.args.eval_hl:
            # when incorporating Commander, both teams are on same level.
            self.args.eval_level_ag = self.args.eval_level_opp = 5

        self.args.eval = True if self.args.render else self.args.eval

        self.args.total_num = self.args.num_agents + self.args.num_opps
        self.args.env_config = {"args": self.args}

    @property
    def get_arguments(self):
        return self.args

