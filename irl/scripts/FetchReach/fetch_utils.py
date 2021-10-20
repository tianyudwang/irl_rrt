import gym
from gym.wrappers import FilterObservation, FlattenObservation
from gym.envs.robotics.utils import robot_get_obs

import numpy as np

from irl.wrapper.fixGoal import FixGoal


def flatten_fixed_goal(env: gym.Env) -> gym.Env:
    """
    Filter and flatten observavtion from Dict to Box and set a fix goal state
    Before:
        obs:
            {
            'observation': array([...]]),  # (n,) depend on env 10 in FetchReach-v1
            'achieved_goal': array([...]), # (3,) # xyz pos of achieved position
            'desired_goal': array([...])  # (3,) # xyz pos of true goal position
            }
    After:
        obs:{
            "":
            "obseravtion": [desired_goal, grip_pos, gripper_state, grip_velp, gripper_vel]
                           [   (3,)       (3,)         (2,)        (3,)       (2,)   ]

            grip_pos = self.sim.data.get_site_xpos("robot0:grip")
            robot_qpos, robot_qvel = robot_get_obs(self.sim)
            gripper_state = robot_qpos[-2:]
            grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
            gripper_vel = robot_qvel[-2:] * dt
        }
    :param: env
    :return flattend env where obs space in Box
    """

    # Filter the observation Dict
    env = FilterObservation(env, ["observation", "desired_goal"])

    # Convert Dict space to Box space
    env = FlattenObservation(env)

    # Fix the goal postion
    env = FixGoal(env)  # custom wrapper might need to double check

    # Sanity Check
    obs = env.reset()
    envGoal = env.goal.copy()

    grip_pos = env.sim.data.get_site_xpos("robot0:grip")

    robot_qpos, robot_qvel = robot_get_obs(env.sim)
    gripper_state = robot_qpos[-2:]

    grip_velp = env.sim.data.get_site_xvelp("robot0:grip") * env.dt
    gripper_vel = robot_qvel[-2:] * env.dt

    verify_obs = np.concatenate(
        [envGoal, grip_pos, gripper_state, grip_velp, gripper_vel], dtype=np.float32
    )
    assert np.all(obs == verify_obs)
    return env


def visulallzie_env(env: gym.Env, joint_idx: int) -> None:
    """
    Visualize the environment
    :param env:
    :param render:
    :param render_video:
    :return:
    """
    while True:
        try:
            env.render()
            env.sim.data.qpos[joint_idx] += 0.01
        except KeyboardInterrupt:
            break
