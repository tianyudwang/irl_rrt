
# Changes made to train.py

## 1

    change the save dir to /irl_rrt/models/

## 2

add an uuid to experiment save folder to avoid conflicts

e.g:

    logdir = args.env_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S") + "_" + f"{uuid.uuid4().hex[:3]}"

## 3. in trainer.init_env()

    add PointUMaze env in trainer.init_env()

## 4

handle different sb3 agent in collect_demo_trajectories()

supress stableBasline3 warnings if running python 3.8+

change paths of  all expert models to /irl_rrt/rl-trained-agents/<expert_name>

e.g:
    /irl_rrt/rl-trained-agents/SAC_Pendulum-v0'

## 5

apply remove time feature wrapper in pointUmaze env since it has a buit in time.

## 6

add a new parser argument to choose which planner to use

# Changes made to irl_agent.py

## 1 in __init__()

differnt planner for 3 environments. I'm not sure which script is for NavEnv-v0 so I left it blank

## 2 env_name

add a new argument env_name to IRL_Agent in order to distinguish 3 differnt environments and planner

## 3 self.train_reward()

In line 126, the enviroment is not wrapped so ther is no env.one_step_transition(ob, agent_ac)

    # ? why not call env.step(agent_ac) instead?
    # agent_next_ob = self.env.one_step_transition(ob, agent_ac)
    agent_next_ob, *_, = self.env.step(agent_ac)

# changes to base_planner_PointUMaze.py

## 1. IRLCostObjective()

I belive the state is 6 dimention correct me if I'm wrong.

## 2. PointUMaze with control plan

I test the entire training with rrt and sst. Both of them receive an invalid start state Error after a while

```
Info:    Found solution with cost 29.60
Info:    SST: Created 5282 states in 7105 iterations
Info:    Solution found in 5.002142 seconds
Warning: SST: Skipping invalid start state (invalid state)
         at line 257 in /home/yiw084/Documents/Github/irl_rrt/ompl-1.5.2/src/ompl/base/src/Planner.cpp
Debug:   SST: Discarded start state Compound state [
Compound state [
RealVectorState [1.79105 1.54257]
SO2State [0.239666]
]
RealVectorState [-1.26495 -0.586646 -1.01255]
]

Error:   SST: There are no valid initial states!
         at line 228 in /home/yiw084/Documents/Github/irl_rrt/ompl-1.5.2/src/ompl/control/planners/sst/src/SST.cpp
Info:    No solution found after 0.001138 seconds

File "/home/yiw084/Documents/Github/irl_rrt/irl/agents/base_planner_PointUMaze.py", line 473, in control_plan
    raise ValueError("OMPL is not able to solve under current cost function")
ValueError: OMPL is not able to solve under current cost function
```