import gym 
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from ompl import base as ob
from ompl import control as oc

from irl.utils.wrappers import IRLEnv, ReacherWrapper
from irl.utils import planner_utils
from irl.rewards.reward_net import RewardNet
from irl.planners import base_planner as bp
from irl.planners import geometric_planner as gp 
from irl.planners import control_planner as cp


np.set_printoptions(precision=3)


def test_reacher_StatePropagator():
    env = gym.make("Reacher-v2")
    env = ReacherWrapper(env)

    planner = cp.ReacherSSTPlanner(env.unwrapped)
    state_propagator = planner.state_propagator

    for i in range(1000):
        # gym step
        obs = env.reset()
        gym_state_before = obs[:6]
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        gym_state_after = obs[:6]   

        # ompl propagate
        state_before = planner.get_StartState(gym_state_before)
        state_after = ob.State(planner.space)
        control = planner.cspace.allocControl()
        action = action.astype(np.float64)
        control[0] = action[0]
        control[1] = action[1]
        state_propagator.propagate(state_before(), control, 2.0, state_after()) 

        # compare
        ompl_state_after = planner_utils.convert_ompl_state_to_numpy(state_after())
        assert np.linalg.norm(ompl_state_after - gym_state_after) < 1e-2, (
            f"States do not match after ompl propagte \n"
            f"gym state {gym_state_after} \n"
            f"ompl state {ompl_state_after}"
        )


def test_reacher_RRTstar_planner():

    env_name = "Reacher-v2"
    env = gym.make(env_name)
    env = ReacherWrapper(env)

    planner = gp.ReacherRRTstarPlanner()
    for _ in range(10):
        obs = env.reset()
        start = obs[:-2].astype(np.float64)
        target = obs[-2:].astype(np.float64)
        status, states, controls = planner.plan(start=start, goal=target)

        print(len(states))

        finger_pos = states[-1][-2:]
        dist = np.linalg.norm(target - finger_pos)
        assert dist <= 0.05, (
            f"Reacher finger position {states[-1]} does not reach target at {target}",
            f"Distance to target is {dist}"
        )


def test_reacher_SST_planner():
    """
    Path returned by planner does not match exactly the rollout path from controls
    """
    env_name = "Reacher-v2"
    env = gym.make(env_name)
    env = ReacherWrapper(env)

    planner = cp.ReacherSSTPlanner(env.unwrapped)
    for _ in range(10):
        obs = env.reset()
        print(obs)
        qpos = env.unwrapped.sim.data.qpos.flat[:]
        qvel = env.unwrapped.sim.data.qvel.flat[:]
        start = obs[:-2].astype(np.float64)
        target = obs[-2:].astype(np.float64)
        status, states, controls = planner.plan(start=start, goal=target)

        print(len(states), len(controls))
        finger_pos = states[-1][-2:]
        dist = np.linalg.norm(target - finger_pos)
        # assert dist <= 0.1, (
        #     f"Reacher state {states[-1]} does not reach target at {target}",
        #     f"Distance to target is {dist}"
        # )

        obs = env.reset()
        env.set_state(qpos, qvel)
        obs = env._get_obs()
        print(obs)
        # env.render()
        rollout_states = [obs]
        for j in range(len(controls)):
            control = controls[j]
            obs, _, _, _ = env.step(control)
            # env.render()
            rollout_states.append(obs)
        rollout_states = np.array(rollout_states)

        import ipdb; ipdb.set_trace()
        assert len(rollout_states) == len(states)
        assert np.linalg.norm(rollout_states[:, :6] - states) < 0.1


def test_env(seed):

    env = gym.make("Reacher-v2")
    env = ReacherWrapper(env)

    rng = np.random.RandomState(seed)
    env_seed = rng.randint(0, (1 << 31) - 1)
    env.seed(env_seed)

    import ipdb; ipdb.set_trace()

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
    # for i in range(10):
        # test_no_reward(i)
        # test_env(i)
    test_reacher_RRTstar_planner()
    # test_reacher_SST_planner()
    # test_planner()
    # test_reacher_StatePropagator()