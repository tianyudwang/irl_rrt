import argparse
import gym
import numpy as np



def build_env(env_name):
    """
    Make env and add env wrappers
    """
    if env_name == 'Reacher-v2':
        env = gym.make(env_name)
        from irl.utils.wrappers import ReacherWrapper
        env = ReacherWrapper(env)
    else:
        raise ValueError('Environment {} not supported yet ...'.format(env_name))
    return env

def train_policy(env, algo, resume_training, policy_name, 
                 timesteps=50000):
    """
    Train the expert policy in RL
    """
    if algo == 'SAC':
        from stable_baselines3 import SAC
        if resume_training:
            model = SAC.load(policy_name)
        else:
            model = SAC("MlpPolicy", env, verbose=1)

        from stable_baselines3.common.logger import configure
        tmp_path = "/tmp/sb3_log/"
        # set up logger
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        model.learn(total_timesteps=timesteps, log_interval=4)
        model.save(policy_name)
    else:
        raise ValueError('RL algorithm {} not supported yet ...'.format(algo))
    return model


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
    parser.add_argument('--env_name', type=str, default='Reacher-v2')
    parser.add_argument('--algo', type=str, default='SAC')
    parser.add_argument('--resume_training', action='store_true')
    args = parser.parse_args()

    env = build_env(args.env_name)
    
    policy_name = args.algo + '_' + args.env_name
    model = train_policy(env, args.algo, args.resume_training, policy_name)

    visualize_policy(env, model)

if __name__ == '__main__':

    main()