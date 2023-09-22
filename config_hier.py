import argparse
import os
import datetime

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser(description='MARL Warsim')

        parser.add_argument('--horizon', type=int, default=400, help='Length of horizon')
        parser.add_argument('--sub_steps', type=int, default=20, help='number of low-level steps')

        parser.add_argument('--restore', type=bool, default=False, help='Path to stored model')
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')

        parser.add_argument('--log_name', type=str, default=None, help='Experiment Name, defaults to Commander + date & time.')
        parser.add_argument('--log_path', type=str, default=None, help='Full Path to actual trained model')

        parser.add_argument('--gpu', type=float, default=0)
        parser.add_argument('--obs_dim', type=int, default=41, help='Observation dimension')
        parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel samplers')
        parser.add_argument('--n_agents', type=int, default=3, help='Number of (trainable) agents')
        parser.add_argument('--n_opponents', type=int, default=3, help='Number of opponents')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=200, help='PPO train batch size')
        parser.add_argument('--mini_batch_size', type=int, default=128, help='PPO train batch size')
        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        self.args = parser.parse_args()
        self.set_metrics()
        self.set_env()

    def set_metrics(self):

        self.args.log_name = f'Commander_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'
        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if self.args.restore:
            if self.args.restore_path is None:
                raise NameError('Specify full path to Commander Policy.')

    def set_env(self):
        self.args.env_config = {
            "args": self.args,
            "horizon": self.args.horizon,
            "num_agents": self.args.n_agents,
            "num_opps": self.args.n_opponents,
            "obs_dim": self.args.obs_dim,
            "sub_steps": self.args.sub_steps,
        }

    @property
    def get_arguments(self):
        return self.args

