import argparse
import sys
import os
import pathlib
import random

from tqdm import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import gym
from gym.wrappers import FilterObservation, FlattenObservation

from stable_baselines3 import HerReplayBuffer, SAC, PPO
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper

from irl.wrapper.fixGoal import FixGoal

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noq


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
        help="Reward type 'sparse' or 'dense' used typically in non-HER training",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Fix the goal positon"
    )
    p.add_argument(
        "--train",
        '-t',
        action="store_true"
    )
    
    p.add_argument(
        "--num_steps",
        "-n",
        type=int,
        help="number of timesteps",
        default=65_000,
    )
    p.add_argument("--device", type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                   help="device to be run on")
    p.add_argument("--seed", help="Random Seed", default=42, type=int)
    p.add_argument(
        "--verbose",
        type=int,
        default=1,
    )
    p.add_argument(
        "--render",
        "-r",
        action="store_true",
        help="Render the environment (useful for tests)",
    )
    
    args = p.parse_args()
    return args

def evaluate(model, env, num_episodes=10, seed=42, render=False, deterministic=True):
    """
    Evaluate a RL agent

    :param model: (BaseSB3RLModel object) the RL Agent
    :param env_id: (str) the environment id
    :param num_episode: (int) number of episodes to evaluate it
    :param norender: (bool) whether to render or not the environment
    """
    env.seed(seed)
    all_episode_rewards = []
     
    for _ in tqdm(range(num_episodes), dynamic_ncols=True):
        env.reset()
        episode_rewards = 0
        done = False
        obs = env.reset()
        
        while not done:
            action, _ = model.predict(obs, deterministic)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            is_success = info.get("is_success")
            if is_success:
                qpos_final = env.sim.get_state().qpos
                qvel_final = env.sim.get_state().qvel
            if render:
                try:
                    env.render()
                except KeyboardInterrupt:
                    sys.exit(0)
        all_episode_rewards.append(episode_rewards)
    # eval_env.close()
    
    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print(f"-" * 50)
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes"
    )
    plt.scatter(range(num_episodes), all_episode_rewards)
    plt.show(block=True)
    return qpos_final, qvel_final



if __name__ == "__main__":
    
    args = CLI()

    # parant directory
    path = pathlib.Path(__file__).parent.parent
    
    # Create save directory and log file
    save_dir = path / "rl-trained-agents"
    fname = str(save_dir / f"{args.env_id}_{args.reward_type}_{args.num_steps}")
    
    if args.fix:
        fname += '_fixGoal'
    ic(fname)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create the environment
    env = gym.make(args.env_id)
    env.seed(args.seed)
    # Set the reward type
    env.reward_type = args.reward_type
    if args.fix:
        # Convert Dict space to Box space
        env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
        # Fix the goal postion
        env = FixGoal(env)
        policy_cls = 'MlpPolicy'
        replay_buffer_class = None
        replay_buffer_kwargs = None
    else:
        policy_cls = "MultiInputPolicy"
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=True,
        )
    
    # SB3 utility wrapper
    for wrapper in [Monitor, TimeFeatureWrapper]:
        env = wrapper(env)
    ic(env)
        
    # Training a agent
    if args.train:
        # Create the model
        model_class = TQC    
        model = model_class(
            policy_cls,
            env, 
            learning_rate=7e-4,
            buffer_size=int(1e6),
            learning_starts=5_000,
            batch_size=1024,
            tau=0.05,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
            # HER params
            # replay_buffer_class=replay_buffer_class,
            # replay_buffer_kwargs=replay_buffer_kwargs,
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
        
        model.learn(total_timesteps = int(args.num_steps))
        model.save(fname)
    else:
        # Because it needs access to `env.compute_reward()`
        # HER must be loaded with the env
        model = TQC.load(fname, env=env)

    # Evaluate the trained agent
    qpos_final, qvel_final = evaluate(model, env, num_episodes=20, seed=args.seed, render=args.render, deterministic=True)
    ic(qpos_final, qvel_final)
    
    np.savez(path / 'scripts' / 'goal.npz', q_pos=qpos_final, q_vel=qvel_final)
    
    