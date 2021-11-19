import argparse
import sys
from typing import Optional

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def build_env(env_name: str):
    """
    Make env and add env wrappers
    """
    assert isinstance(env_name, str)
    if env_name == "NavEnv-v0":
        import gym_nav

        env = gym.make(env_name)
    elif env_name == "Pendulum-v0":
        env = gym.make(env_name)
        from pendulum_env_wrapper import PendulumWrapper

        env = PendulumWrapper(env)
    elif env_name in ["PointUMaze-v0", "PointUMaze-v1"]:
        # v0 adn v1 has different reward function, all others are the same
        import mujoco_maze
        from remove_timeDim_wrapper import RemovTimeFeatureWrapper

        # * This env includes the time at the last axis, which should be removed.
        env = RemovTimeFeatureWrapper(gym.make(env_name))
        ic(env.observation_space)
    else:
        raise ValueError("Environment {} not supported yet ...".format(env_name))
    return Monitor(env)


def train_policy(
    env: gym.Env,
    env_name: str,
    algo: str,
    resume_training: bool,
    eval_env: Optional[gym.Env] = None,
    device: str = "auto",
    wrapped_time: bool = False,
    policy_name: Optional[str] = None,
    timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    learning_starts: int = 1_000,
    reward_threshold: Optional[float] = None,
):
    """
    Train the expert policy in RL
    """
    # if wrapped_time:
    #     from sb3_contrib.common.wrappers import TimeFeatureWrapper
    #     # * Note this wrapper will append time to observation which might not work with expert policy
    #     env = TimeFeatureWrapper(env)
    #     print("Wrapping the env in a TimeFeatureWrapper")

    callback = []

    eval_callback_kwargs = dict(
        eval_env=eval_env,
        best_model_save_path=f"./logs/{env_name}/{algo}/best_model",
        log_path=f"./logs/{env_name}/{algo}/results",
        n_eval_episodes=5,
        eval_freq=10_000,
        verbose=1,
    )

    if reward_threshold is not None:
        assert isinstance(reward_threshold, (float, int))
        print(
            f"Stop training when the model reaches the reward threshold: {reward_threshold}"
        )
        # Stop training when the model reaches the reward threshold
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold, verbose=1
        )
        eval_callback_kwargs["callback_on_new_best"] = callback_on_best

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(**eval_callback_kwargs)
    callback.append(eval_callback)

    if callback:
        # Create the callback list if it is not empty
        callback = CallbackList(callback)

    algo = algo.upper()
    if algo in ["SAC", "PPO", "TQC"]:
        if algo == "SAC":
            from stable_baselines3 import SAC

            algo_cls = SAC

        elif algo == "PPO":
            from stable_baselines3 import PPO

            algo_cls = PPO

        elif algo == "TQC":
            try:
                from sb3_contrib import TQC
            except BaseException:
                raise ValueError(
                    "sb3_contrib not installed: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib"
                )
            algo_cls = TQC
    else:
        raise ValueError(f"RL algorithm {algo} not supported yet ...")

    if resume_training:
        print("Resuming training ...\n")
        model = algo_cls.load(policy_name, device=device)
    else:
        print("Training from scratch ...\n")
        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            device=device,
            verbose=1,
            learning_starts=int(learning_starts),
            # gradient_steps=64,
            # policy_kwargs=dict(log_std_init=-2, net_arch=[64, 64]),
            # train_freq=64,
            # use_sde=True,
        )
    model.learn(total_timesteps=int(timesteps), log_interval=4, callback=callback)

    return model


def save_policy(model, policy_name):
    model.save(policy_name)


def load_policy(algo, model_path, env, device="cpu"):
    algo = algo.upper()
    if algo == "SAC":
        from stable_baselines3 import SAC

        algo_cls = SAC

    elif algo == "PPO":
        from stable_baselines3 import PPO

        algo_cls = PPO

    elif algo == "TQC":
        try:
            from sb3_contrib import TQC
        except BaseException:
            raise ValueError(
                "sb3_contrib not installed: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib"
            )
        algo_cls = TQC
    else:
        raise ValueError(f"RL algorithm {algo} not supported yet ...")

    ic(model_path)
    model = algo_cls.load(model_path, env=env, device=device)
    return model


def visualize_policy(env, model, num_episodes=10, render=True):
    """
    Visualize the policy in env
    """
    # Ensure testing on same device
    total_ep_returns = []
    total_ep_lengths = []

    for _ in range(num_episodes):
        obs = env.reset()
        ep_ret, ep_len = 0.0, 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_ret += reward
            ep_len += 1

            if render:
                try:
                    env.render()
                except KeyboardInterrupt:
                    sys.exit(0)
            if done:
                total_ep_returns.append(ep_ret)
                total_ep_lengths.append(ep_len)
                obs = env.reset()

    mean_episode_reward = np.mean(total_ep_returns)
    std_episode_reward = np.std(total_ep_lengths)
    print(f"-" * 50)
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes"
    )
    print(f"-" * 50)
    env.close()
    return total_ep_returns


def main():
    parser = argparse.ArgumentParser(description="Train the expert policy in RL")
    parser.add_argument(
        "--env_name",
        type=str,
        choices=["NavEnv-v0", "Pendulum-v0", "PointUMaze-v0", "PointUMaze-v1"],
        required=True,
    )
    parser.add_argument("--algo", type=str, default="SAC")
    parser.add_argument("--num_steps", "-n", type=int, default=100_000)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    parser.add_argument("--learning_starts", "-start", type=float, default=10_000)
    parser.add_argument("--reward_threshold", type=float)

    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--timeWrapper", "-wrap", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--train", "-t", action="store_true")
    parser.add_argument("--render", "-r", action="store_true")
    args = parser.parse_args()

    env = build_env(args.env_name)
    eval_env = build_env(args.env_name)

    policy_name = args.algo.upper() + "_" + args.env_name

    if args.train:
        print(
            f"Training {args.algo.upper()} on {args.env_name} with lr = {args.learning_rate}"
        )
        model = train_policy(
            env,
            args.env_name,
            args.algo,
            args.resume_training,
            eval_env=eval_env,
            policy_name=policy_name,
            device="cuda" if args.cuda else "cpu",
            wrapped_time=args.timeWrapper,
            timesteps=args.num_steps,
            learning_rate=args.learning_rate,
            learning_starts=args.learning_starts,
            reward_threshold=args.reward_threshold,
        )
        save_policy(model, policy_name)

    # Load policy from file

    model_path = f"./logs/{args.env_name}/{args.algo.lower()}/best_model/best_model"
    model = load_policy(args.algo, model_path, env, device="cpu")

    # Evaluate the policy and Optionally render the policy
    total_ep_returns = visualize_policy(env, model, render=args.render)
    plt.scatter(range(len(total_ep_returns)), total_ep_returns)
    plt.show()


if __name__ == "__main__":

    main()
