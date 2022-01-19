import gym
import numpy as np

from irl.utils.wrappers import PendulumWrapper


def test_one_step_transition():
    seed = 0
    rng = np.random.RandomState(seed)
    env_seed = rng.randint(0, (1 << 31) - 1)
    env = PendulumWrapper(gym.make("Pendulum-v1"))
    env.seed(int(env_seed))

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
    seed = 0
    rng = np.random.RandomState(seed)
    env_seed = rng.randint(0, (1 << 31) - 1)
    env = PendulumWrapper(gym.make("Pendulum-v1"))
    env.seed(int(env_seed))

    high = np.array([np.pi, 8])

    for _ in range(100):
        reset_location = np.random.uniform(low=-high, high=high)
        obs = env.reset(reset_location)
        assert np.allclose(obs, reset_location)


if __name__ == '__main__':
    test_one_step_transition()