import time
from math import cos, sin

import numpy as np
import gym
import matplotlib.pyplot as plt

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noq


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    obs = env.reset()

    for i in range(200):
        env.render()

        obs, *_ = env.step([-2])
        cos_theta, sin_theta, theta_dot = obs
        y = cos_theta
        x = -sin_theta

        th_rad = np.arctan2(x, y)
        th_deg = np.rad2deg(np.arctan2(x, y))
        ic(x, y, theta_dot, th_rad, th_deg)
        if not np.allclose(cos(th_rad), y):
            print(f"ERROR in y:{y}, {cos(th_rad)}")
        if not np.allclose(sin(th_rad), x):
            print(f"ERROR in x:{x}, {sin(th_rad)}: {x-sin(th_rad)}")

        plt.plot(x, y, "ro")
        plt.plot([0, x], [0, y])
        plt.plot([0, 0], [1, 0], "--")
        plt.scatter(x=0, y=0)
        plt.scatter(x=0, y=1, c="g")

        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.show()
        try:
            time.sleep(0.5)
        except KeyboardInterrupt:
            break

        x_test = sin(th_rad)
        print(f"x_test: {sin_theta - x_test}")
