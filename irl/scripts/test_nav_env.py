import gym 
import gym_nav 
from stable_baselines3 import SAC

class NavIRLEnv(gym.Wrapper):
    def __init__(self, env, reward):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.reward = reward

    def step(self, action):
        """
        Override the true environment reward with learned reward
        """
        obs, reward, done, info = self.env.step(action)
        #reward = self.reward(self.last_obs, obs).item()
        self.last_obs = obs 
        return obs, reward, done, info

    def reset(self):
        self.last_obs = self.env.reset()
        return self.last_obs

def train_policy():
    env = gym.make('NavEnv-v0') 
    model = SAC("MlpPolicy", env, verbose=1)
    #model.load("SAC_NavEnv-v0")
    model.learn(total_timesteps=10000, log_interval=4)
    # Save model
    model.save("SAC_NavEnv-v0")
    
    # visualize        
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

def reward_fn(state, next_state):
    return 0

def test_env():
    env = gym.make('NavEnv-v0')
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000, log_interval=4)

    obs = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)


    irl_env = NavIRLEnv(env, reward_fn)
    model = SAC("MlpPolicy", irl_env, verbose=1)
    model.learn(total_timesteps=1000, log_interval=4)

    obs = irl_env.reset()
    for i in range(10):
        action = irl_env.action_space.sample()
        obs, reward, done, info = irl_env.step(action)
        print(obs, reward, done, info)



def main():
    #train_policy()
    
    test_env()




if __name__ == '__main__':
    main()