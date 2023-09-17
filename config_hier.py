import argparse
import os
import json

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser(description='MARL Warsim')

        parser.add_argument('--persistence', type=bool, default=False, help='if to use persistence Json file - if not, allows for parralel sampling')
        parser.add_argument('--sequential', type=bool, default=True, help='Architecture Type')
        parser.add_argument('--global_policy', type=bool, default=False, help='Commander Policy Type')

        parser.add_argument('--horizon', type=int, default=400, help='Length of horizon')
        parser.add_argument('--sub_steps', type=int, default=20, help='number of low-level steps')
        parser.add_argument('--ss', type=int, default=2, help='Observation type')
        parser.add_argument('--opp_selection', type=bool, default=True, help='Opp Selection')

        parser.add_argument('--restore', type=bool, default=False, help='Path to stored model')
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')

        parser.add_argument('--gpu', type=float, default=0)
        parser.add_argument('--obs_dim', type=int, default=41, help='Observation dimension')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel samplers')
        parser.add_argument('--n_agents', type=int, default=3, help='Number of (trainable) agents')
        parser.add_argument('--n_opponents', type=int, default=3, help='Number of opponents')
        parser.add_argument('--max_n', type=int, default=3, help='max number of agents')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=800, help='PPO train batch size')
        parser.add_argument('--mini_batch_size', type=int, default=128, help='PPO train batch size')
        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        self.args = parser.parse_args()
        self.set_metrics()
        self.set_env()
        if self.args.persistence:
            self.set_persistence()

    def set_metrics(self):

        exp_name = 'HIER_Gru_Adj1'
        #self.args.exp_folder = f'{exp_name}'

        self.args.log_name = f'{exp_name}'

        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if self.args.restore:
            if self.args.restore_path is None:
                self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', 'HIER_GruSel_Pre', 'checkpoint')

        if self.args.ss == 2 and not self.args.global_policy:
            self.args.obs_dim = 41

        if self.args.global_policy:
            if self.args.ss == 1:
                self.args.obs_dim = 150
            else:
                self.args.obs_dim = 55

        if self.args.persistence:
            self.args.num_workers = 0

    def set_env(self):
        self.args.env_config = {
            "args": self.args,
            "horizon": self.args.horizon,
            "num_agents": self.args.n_agents,
            "num_opps": self.args.n_opponents,
            "max_n": self.args.max_n,
            "obs_dim": self.args.obs_dim,
            "sequential": self.args.sequential,
            "sub_steps": self.args.sub_steps,
            "ss": self.args.ss,
            "opp_selection": self.args.opp_selection
        }

    def set_persistence(self):
        if not os.path.exists(os.path.join('results', self.args.log_name, 'checkpoint')):
            os.makedirs(os.path.join('results', self.args.log_name, 'checkpoint'))

        esc_wait = 0 if self.args.opp_mode == "escape" else 3

        data = {"level": self.args.level, "acc_reward":0, "reward_mean": 0, "ss": self.args.ss, "iteration": 0, "epoch": 0, "level_epoch":0, "friendly_kills": 0, "opponent_kills": 0, "got_killed":0, "sp_update": False, "sp_memory": False, "escape_wait": esc_wait}
        with open('results/' + self.args.log_name + "/metrics.json", "w") as file:
            json.dump(data, file)

    @property
    def get_arguments(self):
        return self.args

