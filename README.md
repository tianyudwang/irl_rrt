# Inverse reinforcement learning using sampling based planning

## Environments
We consider two mujoco tasks: `maze2d-umaze-v0` and `antmaze-umaze-v0` from [d4rl](https://github.com/rail-berkeley/d4rl)

## Requirements
1. Python 3.8+

2. Install [OMPL](https://ompl.kavrakilab.org/) with python bindings.
Add python bindings to PYTHONPATH
```bash
export PYTHONPATH=path_to_ompl/py-bindings
```

3. Install [d4rl](https://github.com/rail-berkeley/d4rl)

4. Install irl package
```bash
pip install -e .
```

## Instructions
```bash
cd irl/scripts
python3 train.py
```
