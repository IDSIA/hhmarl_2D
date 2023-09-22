# HHMARL

This is the official implementation of the method proposed in the paper "Hierarchical Multi-Agent Reinforcement Learning
for Air Combat Maneuvering". 

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

Run `train_cc.py` for homogeneous agent trainig, `train_hetero.py` for heterogeneous agents training and `train_hier.py` to train the super-policy (commander). 

`config.py` contain the corresponding arguments to set for training, e.g. `batch_size`. 
`agent_mode` can be `"fight"` or `"escape"`. `render` stores iteratively the current combat situation as .png file (for a video-like visualization, best to open the generated file `current.png` in VSCode while running the code). 

At this stage, low-level policy training is configured for 2vs2 only. High-level commander policy accepts any combat configuration.

`play.py` allows to try out the simulation environment with as many agents as desired. 






