import argparse
import sys
import os
import pathlib
import time
import random

from tqdm import tqdm

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import gym
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3 import HerReplayBuffer, SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper


try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noq


class FixGoal(gym.Wrapper):
    def __init__(self, env, pos=(1.3040752, 0.74440193, 0.66095406)):
        super().__init__(env)
        self.env = env
        assert len(pos) == 3, "pos should be a list of 3 elements representing x, y, z positions"
        if not isinstance(pos, np.ndarray):
            pos = np.asarray(pos, dtype=np.float32)
        self.pos = pos

    def step(self, action):
        observation, _, done, info = self.env.step(action)
        achieved_goal = observation[3:6]
        reward = self.compute_reward(achieved_goal, self.env.goal)

        return observation, reward, done, info

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert isinstance(goal_a, np.ndarray) and isinstance(goal_b, np.ndarray)
        assert goal_a.shape == goal_b.shape
        assert len(goal_a) == len(goal_b) == 3
        
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info=None):
        d = self.goal_distance(achieved_goal, goal)
        if self.env.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


    def reset(self):
        obs = self.env.reset()
        self.env.goal[0] = self.pos[0]
        self.env.goal[1] = self.pos[1]
        self.env.goal[2] = self.pos[2]

        # ! The following line does not work
        # self.env.goal = self.pos
        obs[0:3] = self.env.goal.copy()
        return obs


def CLI():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id", type=str, help="environment ID", default="FetchReach-v1" 
    )
    p.add_argument(
        "--reward_type",
        "-rt",
        type=str,
        choices=["dense", "sparse"],
        default="dense",
        help="Reward type 'sparse' or 'dense' used in non-HER training ",
    )
    p.add_argument(
        "--num_steps",
        "-n",
        type=int,
        help="number of timesteps",
        default=80_000,
    )
    p.add_argument("--device", type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                   help="device to be run on")
    p.add_argument("--seed", help="number of timesteps", default=42, type=int)
    p.add_argument(
        "--verbose",
        type=int,
        default=1,
    )
    p.add_argument(
        "-nr",
        "--norender",
        action="store_true",
        default=False,
        help="Do not render the environment (useful for tests)",
    )
    
    args = p.parse_args()
    return args

def evaluate(model, env_id, num_episodes=10, seed=42, render=False, deterministic = True):
    """
    Evaluate a RL agent

    :param model: (BaseSB3RLModel object) the RL Agent
    :param env_id: (str) the environment id
    :param num_episode: (int) number of episodes to evaluate it
    :param norender: (bool) whether to render or not the environment
    """
    eval_env = gym.make(env_id)
    eval_env.seed(seed)
    all_episode_rewards = []
    for _ in tqdm(range(num_episodes)):
        eval_env.reset()
        episode_rewards = 0
        done = False
        obs = eval_env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic)
            obs, reward, done, info = eval_env.step(action)
            episode_rewards += reward
            if render:
                try:
                    eval_env.render()
                except KeyboardInterrupt:
                    sys.exit(0)
        all_episode_rewards.append(episode_rewards)
    eval_env.close()
    
    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print(f"-" * 50)
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes"
    )
    plt.plot(all_episode_rewards)
    plt.show()


if __name__ == "__main__":
    
    args = CLI()

    # parant directory
    path = pathlib.Path(__file__).parent.parent
    
    save_dri = path / "rl-trained-agents"
    
    ALGO = {
        "ppo": PPO,
        "sac": SAC,
        "tqc": TQC,
    }
    
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create the environment
    env = gym.make(args.env_id)
    env.seed(args.seed)
    # Set the reward type
    env.reward_type = args.reward_type
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
    env = FixGoal(env)
    env = Monitor(env)
    env = TimeFeatureWrapper(env)
    
    ic(env)
    
    model_class = SAC    
    
   
    model = TQC(
        "MlpPolicy",
        env, 
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=5_000,
        batch_size=1024,
        tau=0.05,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        # replay_buffer_class=HerReplayBuffer,
        # replay_buffer_kwargs=dict(
        #     n_sampled_goal=4,
        #     goal_selection_strategy='final',
        #     online_sampling=True,
        # ),
        optimize_memory_usage=False,
        ent_coef='auto',
        target_update_interval=1,
        target_entropy='auto',
        top_quantiles_to_drop_per_net=2,
        use_sde=False,
        sde_sample_freq=- 1,
        use_sde_at_warmup=False,
        tensorboard_log=None,
        create_eval_env=False,
        policy_kwargs=None,
        verbose=args.verbose, 
        seed=args.seed,
        device=args.device,
    )
    
    model.learn(total_timesteps=int(args.num_steps))
    
    fname = str(save_dri / f"{args.env_id}_{args.reward_type}_{args.num_steps}")
    model.save(fname)
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    # model = model_class.load(fname, env=env)
    
    
    # Evaluate the trained agent
    