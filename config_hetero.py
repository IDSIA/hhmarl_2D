import argparse
import os
import json

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser(description='MARL Warsim')

        parser.add_argument('--curriculum', type=bool, default=False, help='If training in Curriculum Mode')
        parser.add_argument('--persistence', type=bool, default=False, help='if to use persistence Json file - if not, allows for parralel sampling')
        parser.add_argument('--agent_mode', type=str, default="fight", help='Agent mode: Fight or Escape')
        parser.add_argument('--opp_mode', type=str, default="fight", help='Opponent mode: Fight or Escape')
        parser.add_argument('--log_history', type=int, default=False, help='Saving history')
        parser.add_argument('--heterogeneous', type=bool, default=True, help='if heterogeneous jets used, fast and slow instances')
        parser.add_argument('--ac_type', type=int, default=1, help='Aircraft type')
        parser.add_argument('--rew_type', type=str, default='const', help='Reward type')
        parser.add_argument('--sequential', type=bool, default=True, help='Reward type')

        parser.add_argument('--horizon', type=int, default=150, help='Length of horizon')
        parser.add_argument('--level', type=int, default=5, help='Length of horizon')
        parser.add_argument('--ss', type=float, default=2, help='State space')

        parser.add_argument('--restore', type=bool, default=True, help='Path to stored model')
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')
        parser.add_argument('--log_name', type=str, default=None, help='Path to actual trained model')
        parser.add_argument('--log_path', type=str, default=None, help='Full Path to actual trained model')
        parser.add_argument('--exp_folder', type=str, default=None, help='Folder Name of experiment')

        parser.add_argument('--gpu', type=float, default=0)
        parser.add_argument('--obs_dim', type=int, default=22, help='Observation dimension')
        parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel samplers')
        parser.add_argument('--n_agents', type=int, default=2, help='Number of (trainable) agents')
        parser.add_argument('--n_opponents', type=int, default=2, help='Number of opponents')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=2000, help='PPO train batch size')
        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        self.args = parser.parse_args()
        self.set_metrics()
        self.set_env()
        if self.args.persistence:
            self.set_persistence()

    def set_metrics(self):

        exp_name = '_HETERO_Att'
        self.args.exp_folder = f'SS{self.args.ss}{exp_name}'

        self.args.log_name = f'SS{self.args.ss}_L{self.args.level}{exp_name}'
        #self.args.log_name = f'Check'

        if self.args.ss == 2:
            self.args.obs_dim = 30

        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', 'HETERO', self.args.exp_folder, self.args.log_name) if self.args.exp_folder else os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if self.args.restore:
            if self.args.restore_path is None:
                if self.args.exp_folder:
                    self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results','HETERO', self.args.exp_folder, f'SS{self.args.ss}_L{self.args.level-1}{exp_name}', 'checkpoint')
                else:
                    self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', 'HETERO', f'SS{self.args.ss}_L{self.args.level-1}{exp_name}', 'checkpoint')

        if self.args.persistence:
            self.args.num_workers = 0

        horizons = {1: 150, 2:200, 3:300, 4:300, 5:400}
        self.args.horizon = horizons[self.args.level]

    def set_env(self):
        self.args.env_config = {
            "args": self.args,
            "horizon": self.args.horizon,
            "num_agents": self.args.n_agents,
            "num_opps": self.args.n_opponents,
            "restore_path": self.args.restore_path,
            "log_path": self.args.log_path,
            "log_name": self.args.log_name,
            "level": self.args.level,
            "ss": self.args.ss,
            "obs_dim": self.args.obs_dim,
            "curriculum": self.args.curriculum,
            "persistence": self.args.persistence,
            "agent_mode": self.args.agent_mode,
            "opp_mode": self.args.opp_mode,
            "ac_type": self.args.ac_type,
            "rew_type": self.args.rew_type,
            "hetero": self.args.heterogeneous,
            "sequential": self.args.sequential
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

