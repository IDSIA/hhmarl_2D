"""
This file exports policy models to use during inference / self-play.
"""

import os
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from models.ac_models_hetero import Esc1, Esc2, Fight1, Fight2

#define experiment folder name
LEVEL = 3
MODE = 'fight'
EXP_DIR = f'Level{LEVEL}_{MODE}'

#define policy folder name
POL_DIR = 'policies'

if MODE == "fight":
    ModelCatalog.register_custom_model(f"ac1_model",Fight1)
    ModelCatalog.register_custom_model(f"ac2_model",Fight2)
else:
    ModelCatalog.register_custom_model(f"ac1_model_esc",Esc1)
    ModelCatalog.register_custom_model(f"ac2_model_esc",Esc2)

for i in range(1,3):
    check = os.path.join(os.path.dirname(__file__), 'results', EXP_DIR, 'checkpoint', 'policies', f'ac{i}_policy')
    pol = Policy.from_checkpoint(check)
    save_dir = os.path.join(os.path.dirname(__file__), POL_DIR)
    pol.export_model(save_dir)
    
    policy_name = f'L{LEVEL}_AC{i}' if MODE == "fight" else f'Esc_AC{i}'
    os.rename(f'{POL_DIR}/model.pt', f'{POL_DIR}/{policy_name}.pt')

print(f"{MODE} policies exported to folder {POL_DIR}")