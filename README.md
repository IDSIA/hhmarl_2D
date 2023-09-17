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




