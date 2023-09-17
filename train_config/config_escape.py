import argparse
import os
import json

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser(description='MARL Warsim')

        parser.add_argument('--horizon', type=int, default=150, help='Length of horizon')
        parser.add_argument('--level', type=int, default=1, help='Length of horizon')
        parser.add_argument('--ss', type=int, default=1, help='State space')
        parser.add_argument('--reward_type', type=int, default=1, help='Reward number')

        parser.add_argument('--restore', type=bool, default=False, help='Path to stored model')
        parser.add_argument('--eval', type=bool, default=False, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')
        parser.add_argument('--log_name', type=str, default=None, help='Path to actual trained model')
        parser.add_argument('--log_path', type=str, default=None, help='Full Path to actual trained model')
        parser.add_argument('--curriculum', type=bool, default=False, help='If training in Curriculum Mode')

        parser.add_argument('--gpu', type=float, default=1)
        parser.add_argument('--obs_dim', type=int, default=20, help='Observation dimension')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel samplers')
        parser.add_argument('--n_agents', type=int, default=2, help='Number of (trainable) agents')
        parser.add_argument('--n_opponents', type=int, default=2, help='Number of opponents')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=2000, help='PPO train batch size')
        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        self.args = parser.parse_args()
        self.set_paths()
        self.set_env()
        self.set_persistence()

    def set_paths(self):

        self.args.log_name = f'ESC_SS{self.args.ss}_Rew{self.args.reward_type}_Lv{self.args.level}'
        #self.args.log_name = 'CHECK'

        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if self.args.restore:
            if self.args.restore_path is None:
                self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', f'ESC_SS{self.args.ss}_Rew{self.args.reward_type}_Lv{self.args.level-1}', 'checkpoint')
                
        if self.args.ss == 1:
            self.args.obs_dim = 10

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
            "rew": self.args.reward_type,
            "curriculum": self.args.curriculum
        }

    def set_persistence(self):
        if not os.path.exists(os.path.join('results', self.args.log_name, 'checkpoint')):
            os.makedirs(os.path.join('results', self.args.log_name, 'checkpoint'))

        # data = {"level": self.args.level, "acc_reward":0, "reward_mean": 0, "reward_type": self.args.reward_type, "ss": self.args.ss, "iteration": 0, "epoch": 0, "level_epoch":0, "friendly_kills": 0, "opponent_kills": 0, "got_killed":0, "sp_update": False, "sp_memory": False, "escape_wait": 2, "acc_raw_rew": 0, "raw_rew_mean": 0}
        # with open('results/' + self.args.log_name + "/metrics.json", "w") as file:
        #     json.dump(data, file)

    @property
    def get_arguments(self):
        return self.args

