import gym 
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from irl.utils.wrappers import IRLEnv, ReacherWrapper
from irl.rewards.reward_net import RewardNet

def test_env(seed):

    env = gym.make("Reacher-v2")
    env = ReacherWrapper(env)

    rng = np.random.RandomState(seed)
    env_seed = rng.randint(0, (1 << 31) - 1)
    env.seed(env_seed)

    nn_params = {
        'ob_dim': env.observation_space.shape[0],
        'output_size': 1,
        'n_layers': 2,
        'size': 256,
        'activation': 'relu',
        'output_activation': 'sigmoid',
        'learning_rate': 0.003,
    }
    reward = RewardNet(nn_params)
    irl_env = IRLEnv(env, reward)

    tmp_path = f"/tmp/sb3_reacher_random_policy/{seed}"
    # set up logger
    new_logger = configure(tmp_path, ["csv", "tensorboard"])
    model = SAC("MlpPolicy", env, verbose=0, seed=seed)
    # model.set_logger(new_logger)
    # model.learn(total_timesteps=30000, log_interval=32)

    eval(env, model)

def eval(env, model):
    rews = []
    n_eval = 64
    for i in range(n_eval):
        obs = env.reset()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        rews.append(rewards)    

    lengths = [len(rew) for rew in rews]
    returns = [sum(rew) for rew in rews]
    print(f"Reacher-v2 {n_eval} episodes")
    print(f"Episode return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Episode length {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")

def test_no_reward(seed):
    import gym



    rng = np.random.RandomState(seed)
    env_seed = rng.randint(0, (1 << 31) - 1)
    env = gym.make("Reacher-v2")

    import ipdb; ipdb.set_trace()
    print(env_seed)
    env.seed(int(env_seed))

    nn_params = {
        'ob_dim': env.observation_space.shape[0],
        'output_size': 1,
        'n_layers': 2,
        'size': 128,
        'activation': 'relu',
        'output_activation': 'sigmoid',
        'learning_rate': 0.003,
    }
    reward = RewardNet(nn_params)
    irl_env = IRLEnv(env, reward)

    tmp_path = f"/tmp/sb3_reacher_irl_env/{seed}"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model = SAC("MlpPolicy", irl_env, verbose=0)
    model.set_logger(new_logger)
    model.learn(total_timesteps=30000, log_interval=32)

    eval(env, model)

    # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # model = SAC("MlpPolicy", env, verbose=1)
    # model.set_logger(new_logger)
    # model.learn(total_timesteps=50000, log_interval=4)

if __name__ == '__main__':
    for i in range(10):
#        test_no_reward(i)
        test_env(i)
