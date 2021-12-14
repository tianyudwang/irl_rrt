import argparse
import gym
import numpy as np 

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


CUSTOM_OBJECTS = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}

def build_env(env_name):
    """
    Make env and add env wrappers
    """
    if env_name == 'NavEnv-v0':
        import gym_nav
        env = gym.make(env_name)
    elif env_name == 'Pendulum-v0':
        env = gym.make(env_name)
        from irl.scripts.wrapper.pendulum_env_wrapper import PendulumWrapper
        env = PendulumWrapper(env)
    elif env_name == 'PointUMaze-v0':
        import mujoco_maze
        from irl.scripts.wrapper.remove_timeDim_wrapper import RemovTimeFeatureWrapper
        
        env = RemovTimeFeatureWrapper(gym.make(env_name))
        
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
        model = SAC.load(model_name, custom_objects=CUSTOM_OBJECTS)
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
    reward_sum = 0
    while episode < num_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        env.render(mode='human')
        if done:
            episode += 1
            obs = env.reset()
    ic(reward_sum/ num_episodes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='PointUMaze-v0')
    parser.add_argument('--algo', type=str, default='SAC')
    parser.add_argument('--policy_name', type=str, default='./models/SAC_PointUMaze-v0_itr_40.zip')
    args = parser.parse_args()

    env = build_env(args.env_name)

    model = load_policy(env, args.algo, args.policy_name)

    visualize_policy(env, model)


if __name__ == '__main__':

    main()