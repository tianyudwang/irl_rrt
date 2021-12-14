# Inverse Reinforcement Learning using Sampling-based Planning

## Environments
We consider a simple continuous space environment: Pendulum-v0

## Requirements
Install [OMPL](https://ompl.kavrakilab.org/) with python bindings.
```bash
pip install -e .
```
Add OMPL pybindings to PYTHONPATH
```bash
export PYTHONPATH=path_to_ompl/py-bindings
```

## Instructions
```bash
cd irl/scripts
python3 train.py --env_name Pendulum-v0 --exp_name Pendulum-v0 --expert_policy SAC_Pendulum-v0
```
