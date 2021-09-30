import unittest
import gym
from stable_baselines3 import SAC

class Path:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add_step(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

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

def load_policy(model_name):
    model = SAC.load(model_name)
    return model

def collect_paths(env, model, num_episodes=10, render=False):
    
    obs = env.reset()
    paths, path = [], Path()
    episode = 0
    while episode < num_episodes:
        action, _states = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        path.add_step(obs, action, reward, done)
        obs = next_obs

        if render:
            env.render()

        if done:
            episode += 1
            obs = env.reset()
            paths.append(path)

    return paths
    

class TestNavEnv(unittest.TestCase):
    def test(self):
        env = build_env('NavEnv-v0')

        model_name = ('../SAC_NavEnv-v0')
        model = load_policy(model_name)

        paths = collect_paths(env, model)

        for path in paths:
            self.assertEqual(len(path.states), len(path.actions))
            self.assertEqual(len(path.actions), len(path.rewards))
            self.assertEqual(len(path.rewards), len(path.dones))




if __name__ == '__main__':
    unittest.main()