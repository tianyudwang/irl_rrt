import gym
import numpy as np

# import irl.planners.geometric_planner as gp
# import irl.planners.control_planner as cp
# from irl.utils.wrappers import PendulumWrapper
# from irl.utils import planner_utils

def init_env():
    seed = 0
    rng = np.random.RandomState(seed)
    env_seed = rng.randint(0, (1 << 31) - 1)
    env = PendulumWrapper(gym.make("Pendulum-v1"))
    env.seed(int(env_seed))
    return env

def test_one_step_transition():
    env = init_env()

    for i in range(100):
        state = env.reset()
        path_1, path_2, actions = [state], [state], []
        for _ in range(10):
            action = env.action_space.sample()
            state, _, _, _ = env.step(action)
            path_1.append(state)
            actions.append(action)  

        state = path_2[0]
        for action in actions:  
            state = env.one_step_transition(state, action)
            path_2.append(state)    

        for state_1, state_2 in zip(path_1, path_2):
            assert np.allclose(state_1, state_2), (
                f"state 1 {state_1} and state 2 {state_2} does not match"
            )

def test_reset():
    env = init_env()
    high = np.array([np.pi, 8])

    for _ in range(100):
        reset_location = np.random.uniform(low=-high, high=high)
        obs = env.reset(reset_location)
        assert np.allclose(obs, reset_location)

def test_rrtstar():
    env = init_env()
    planner = gp.PendulumRRTstarPlanner()

    for _ in range(10):
        obs = env.reset()

        status, states, _ = planner.plan(obs, solveTime=0.4)

        assert np.linalg.norm(states[-1]) <= 0.1, (
            f"Final state {states[-1]} does not reach goal"
        )
        # planner_utils.visualize_path(states)

def test_sst():

    env = init_env()
    planner = cp.PendulumSSTPlanner()
    goal = np.array([0., 0.])

    for _ in range(10):
        obs = env.reset()

        status, states, controls = planner.plan(obs, solveTime=1.0)

        assert np.linalg.norm(states[-1]) <= 0.1, (
            f"Final state {states[-1]} does not reach goal"
        )
        # planner_utils.visualize_path(states)

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
    print(f"Pendulum-v1 {n_eval} episodes")
    print(f"Episode return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Episode length {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")

def test_env(seed):
    import gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure
    from irl.utils.wrappers import PendulumWrapper, IRLEnv
    from irl.rewards.reward_net import RewardNet

    rng = np.random.RandomState(seed)
    env_seed = rng.randint(0, (1 << 31) - 1)
    env = PendulumWrapper(gym.make("Pendulum-v1"))
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

    tmp_path = f"/tmp/sb3_irl_env_sigmoid/{seed}"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model = SAC("MlpPolicy", env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=30000, log_interval=32)

    eval(env, model)


if __name__ == '__main__':
    # test_one_step_transition()
    # test_rrtstar()
    # test_sst()
    for i in range(10):
        test_env(i)
