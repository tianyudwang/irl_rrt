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
    elif env_name == 'Pendulum-v0':
        env = gym.make(env_name)
        from pendulum_env_wrapper import PendulumWrapper
        env = PendulumWrapper(env)
    else:
        raise ValueError('Environment {} not supported yet ...'.format(env_name))
    return env

def load_policy(
        env, 
        algo='SAC',
        model_name='SAC_NavEnv-v0'):
    """
    Train the expert policy in RL
    """
    if algo == 'SAC':
        from stable_baselines3 import SAC
        model = SAC.load(model_name)
    else:
        raise ValueError('RL algorithm {} not supported yet ...'.format(algo))
    return model


def visualize_policy(env, model, num_episodes=10):
    """
    Visualize the policy in env
    """
    obs = env.reset()
    episode = 0
    step = 0
    while episode < num_episodes:
        action, _states = model.predict(obs, deterministic=True)
        assert -np.pi <= obs[0] <= np.pi, "Theta out of bounds"
        assert -8 <= obs[1] <= 8, "Theta dot out of bounds"
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            episode += 1
            obs = env.reset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='NavEnv-v0')
    parser.add_argument('--algo', type=str, default='SAC')
    parser.add_argument('--policy_name', type=str, default='../models/SAC_NavEnv-v0_itr_19.zip')
    args = parser.parse_args()

    env = build_env(args.env_name)
    import pdb; pdb.set_trace()
    model = load_policy(env, args.algo, args.policy_name)

    visualize_policy(env, model)


if __name__ == '__main__':

    main()