# Generate D4RL Dataset

Download the pre-trained antmaze policy [here](http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_hierarch_pol.pkl):

Loading the pickle file requires installing this fork of [RLkit](https://github.com/aviralkumar2907/rlkit-offline-rl-benchmark), and installing torch 1.5.1 and torchvision 0.6.0. (current version torch==1.10.1 torchvision==0.11.2 works fine)

copy the `localmotion` folder and replace `__init__.py` with empty one and run the following command:

```bash
export PYTHONPATH=path_to_localmotion/
```
