
# Changes made to train.py

## 1

    change the save dir to /irl_rrt/models/

## 2

add an uuid to experiment save folder to avoid conflicts

e.g:

    logdir = args.env_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S") + "_" + f"{uuid.uuid4().hex[:3]}"

## 3

    add PointUMaze env in trainer.init_env()
    ```
    ```

## 4

handle different sb3 agent in collect_demo_trajectories()

supress stableBasline3 warnings if running python 3.8+

change paths of  all expert models to /irl_rrt/rl-trained-agents/<expert_name>

e.g:
    /irl_rrt/rl-trained-agents/SAC_Pendulum-v0'

## 5

add a new argument env_name to IRL_Agent in order to distinguish 3 differnt environments and planner

## 6

apply remove time feature wraooer in pointUmaze env since it has a buit in time.

