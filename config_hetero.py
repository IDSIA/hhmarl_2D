import argparse
import os
import datetime

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser(description='MARL Warsim')

        parser.add_argument('--agent_mode', type=str, default="fight", help='Agent mode: Fight or Escape')
        parser.add_argument('--level', type=int, default=1, help='Length of horizon')

        parser.add_argument('--eval', type=bool, default=False, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')
        parser.add_argument('--restore', type=bool, default=False, help='To restore training from a model')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')
        
        parser.add_argument('--log_name', type=str, default=None, help='Experiment Name, defaults to level + train config')
        parser.add_argument('--log_path', type=str, default=None, help='Full Path to actual trained model')

        parser.add_argument('--gpu', type=float, default=0)
        parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel Ray samplers')
        parser.add_argument('--n_agents', type=int, default=2, help='Number of (trainable) agents')
        parser.add_argument('--n_opponents', type=int, default=2, help='Number of opponents')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=2000, help='PPO train batch size')
        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        self.args = parser.parse_args()
        self.set_metrics()
        self.set_env()

    def set_metrics(self):

        #self.args.log_name = f'Level{self.args.level}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'
        self.args.log_name = f'Level{self.args.level}_{self.args.agent_mode}_{self.args.n_agents}vs{self.args.n_opponents}'
        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if self.args.restore:
            if self.args.restore_path is None:
                self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', f'Level{self.args.level-1}_{self.args.agent_mode}_{self.args.n_agents}vs{self.args.n_opponents}', 'checkpoint')
                
        horizons = {1: 150, 2:200, 3:300, 4:300, 5:400}
        self.args.horizon = horizons[self.args.level]
        self.args.eval = True if self.args.render else self.args.eval

    def set_env(self):
        self.args.env_config = {
            "args": self.args,
            "num_agents": self.args.n_agents,
            "num_opps": self.args.n_opponents,
            "level": self.args.level,
            "agent_mode": self.args.agent_mode,
        }

    @property
    def get_arguments(self):
        return self.args

