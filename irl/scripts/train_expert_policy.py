import argparse
import gym
import numpy as np


def build_env(env_name):
    """
    Make env and add env wrappers
    """
    if env_name == 'NavEnv-v0':
        import gym_nav
        env = gym.make(env_name)
    else:
        raise ValueError('Environment {} not supported yet ...'.format(env_name))
    return env

def train_policy(
        env, 
        algo='SAC', 
        timesteps=100000):
    """
    Train the expert policy in RL
    """
    if algo == 'SAC':
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=timesteps, log_interval=4)
    else:
        raise ValueError('RL algorithm {} not supported yet ...'.format(algo))
    return model

def save_policy(model, policy_name):
    model.save(policy_name)

def visualize_policy(env, model, num_episodes=10):
    """
    Visualize the policy in env
    """
    obs = env.reset()
    episode = 0
    while episode < num_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            episode += 1
            obs = env.reset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='NavEnv-v0')
    parser.add_argument('--algo', type=str, default='SAC')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--policy_name', type=str, default='SAC_NavEnv-v0')
    args = parser.parse_args()

    env = build_env(args.env_name)

    model = train_policy(env, args.algo)

    save_policy(model, args.policy_name)

    visualize_policy(env, model)

if __name__ == '__main__':

    main()