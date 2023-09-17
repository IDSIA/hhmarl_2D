import argparse
import os
import json

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser(description='MARL Warsim')

        parser.add_argument('--gpu', type=float, default=0)
        parser.add_argument('--num_workers', type=int, default=0, help='Number of parallel samplers')
        parser.add_argument('--log_name', type=str, default=None, help='Name of experiment folder')

        parser.add_argument('--mode', type=str, default='', help='Neural Network type for MARL')
        parser.add_argument('--horizon', type=int, default=400, help='Length of horizon')

        parser.add_argument('--restore_path_fight', type=str, default=None, help='Path to stored model')
        parser.add_argument('--restore_path_escape', type=str, default=None, help='Path to stored model')
        parser.add_argument('--restore', type=bool, default=False, help='Path to stored model')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')
        parser.add_argument('--eval', type=bool, default=False, help='Enable evaluation mode')
        parser.add_argument('--eval_only', type=bool, default=False, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')

        parser.add_argument('--n_agents', type=int, default=2, help='Number of (trainable) agents')
        parser.add_argument('--n_opponents', type=int, default=2, help='Number of opponents')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=500, help='PPO train batch size')
        parser.add_argument('--sub_steps', type=int, default=20, help='Number of steps in sub-policies')

        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        parser.add_argument('--obs_dim', type=int, default=20, help='Number of steps in sub-policies')


        self.args = parser.parse_args()
        self.set_paths()
        self.set_env()

    def set_paths(self):
        mode = 'Hier'

        if self.args.eval_only:
            log_path = f'{mode}_v1_EVAL'
        else:
            #log_path = f'{mode}_v1'
            log_path = f'CHECK'

        restore_path_fight = os.path.join(os.path.dirname(__file__), 'results', 'Hierarchy' ,'policy_fight')
        restore_path_escape = os.path.join(os.path.dirname(__file__), 'results', 'Hierarchy' ,'policy_escape')

        if self.args.restore:
            if self.args.restore_path is None:
                self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results',  f'{mode}_v1', 'checkpoint')

        self.args.log_name = log_path
        self.args.restore_path_fight = restore_path_fight
        self.args.restore_path_escape = restore_path_escape
        self.args.mode = mode

    def set_env(self):
        self.args.env_config = {
            "mode": self.args.mode,
            "horizon": self.args.horizon,
            "num_agents": self.args.n_agents,
            "num_opps": self.args.n_opponents,
            "path_fight": self.args.restore_path_fight,
            "path_escape": self.args.restore_path_escape,
            "sub_steps": self.args.sub_steps
        }

    @property
    def get_arguments(self):
        return self.args

