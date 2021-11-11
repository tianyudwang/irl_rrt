import argparse
import sys
from typing import Optional

import gym
import numpy as np
import matplotlib.pyplot as plt


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

        env = gym.make(env_name)
    else:
        raise ValueError("Environment {} not supported yet ...".format(env_name))
    return env


def train_policy(
    env: gym.Env,
    algo: str,
    resume_training: bool,
    device: str = "auto",
    wrapped_time: bool = False,
    policy_name: Optional[str] = None,
    timesteps: int = 100_000,
):
    """
    Train the expert policy in RL
    """
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(env)
    if wrapped_time:
        from sb3_contrib.common.wrappers import TimeFeatureWrapper
        env = TimeFeatureWrapper(env)
        print("Wrapping the env in a TimeFeatureWrapper")
    
    
    if algo.upper() in ["SAC", "PPO", "TQC"]:
        if algo == "SAC":
            from stable_baselines3 import SAC
            algo_cls = SAC

        elif algo == "PPO":
            from stable_baselines3 import PPO
            algo_cls = PPO

        elif algo == "TQC":
            try:
                from sb3_contrib import TQC
            except:
                raise ValueError(
                    "sb3_contrib not installed: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib"
                )
            algo_cls = TQC
    else:
        raise ValueError(f"RL algorithm {algo} not supported yet ...")

    
    if resume_training:
        model = algo_cls.load(policy_name)
    else:
        model = algo_cls("MlpPolicy", env, device=device, verbose=1, learning_starts=100_000)
    model.learn(total_timesteps=int(timesteps), log_interval=4)

    return model


def save_policy(model, policy_name):
    model.save(policy_name)


def visualize_policy(env, model, num_episodes=20, render=True):
    """
    Visualize the policy in env
    """
    # Ensure testing on same device
    model.to("cpu")

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
    plt.scatter(range(num_episodes), total_ep_returns)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        choices=["NavEnv-v0, Pendulum-v0", "PointUMaze-v0", "PointUMaze-v1"],
        required=True,
    )
    parser.add_argument("--algo", type=str, default="SAC")
    parser.add_argument("--num_steps", "-n", type=int, default=100_000)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--timeWrapper", "-wrap", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    env = build_env(args.env_name)

    policy_name = args.algo + "_" + args.env_name
    model = train_policy(
        env,
        args.algo,
        args.resume_training,
        policy_name=policy_name,
        device="cuda" if args.cuda else "cpu",
        wrapped_time=False,
        timesteps = args.num_steps,
    )

    save_policy(model, policy_name)

    visualize_policy(env, model)


if __name__ == "__main__":

    main()
