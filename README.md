# HHMARL

Heterogeneous Hierarchical Multi-Agent Reinforcement Learning for Air Combat Maneuvering, the implementation of the method proposed in this [paper](https://arxiv.org/abs/2309.11247).

![Hierarchy Trajectory](img/hier_pol.png) ![Fight Trajectory](img/fight_pol.png)

## Requiered Packages 

- ray["rllib"] == 2.4.0
- torch >= 2.0.0
- numpy == 1.24.3
- gymnasium == 0.26.3
- tensorboard == 2.13.0
- pycairo == 1.23.0
- cartopy >= 0.21.0
- geographiclib == 2.0

## Training

Run `train_hetero.py` for heterogeneous agents training and `train_hier.py` to train the super-policy (commander). The low-level policies must be trained in order to start training of the commander policy. 

`config.py` contain the corresponding arguments to set:

- `agent_mode` is either "fight" or "escape"
- `level` from 1 to 5
- `restore` either True or False, to restore training
- `log_name` to define the experiment name
- `gpu` either 0 or 1, to use gpu or not
- `n_agents` and `n_opponents`, to specify the number of agents and opponents
- `eval` either True or False, for having evaluations stored as images
- `render` either True or False, to visualize the current combat scenario. It stores iteratively the current combat situation as .png file

At this stage, low-level policy training is configured **for 2vs2 only**. High-level commander policy accepts any combat configuration.






