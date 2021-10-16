import argparse
import sys
import os
import time
from typing import Any, Dict
import random

from math import sin, cos, pi
from functools import partial

from PIL import Image
import imageio

import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

import ompl_utils


try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(
        0, join(dirname(dirname(dirname(abspath(__file__)))), "ompl/py-bindings")
    )
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og


def isStateValid(si: ob.SpaceInformation, state: ob.State) -> bool:
    """perform collision checking or check if other constraints are satisfied"""
    l = 1
    # *Since state is a compound state, we need to access its subcomponents Separately
    temp_SO2State = state[0]
    temp_VectorState = state[1]

    th = temp_SO2State.value
    th_dot = temp_VectorState[0]

    # position valid
    x_state, y_state = sin(th), cos(th)
    pos_inbound = np.hypot(x_state, y_state) == l

    # velocity valid
    vel_inbound = -8 <= th_dot <= 8

    return si.satisfiesBounds(state) and pos_inbound and vel_inbound


def propagate(
    start: ob.State, control: oc.Control, duration: float, state: ob.State
) -> bool:
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    max_speed = 8.0
    max_torque = 2.0

    # *Since state is a compound state, we need to access its subcomponents Separately
    temp_SO2State = start[0]
    temp_VectorState = start[1]

    th = temp_SO2State.value
    th_dot = temp_VectorState[0]
    assert -pi <= th <= pi, f"State theta is out of bounds: {th}"
    assert (
        -max_speed <= th_dot <= max_speed
    ), f"State theta_dot is out of bounds: {th_dot}"

    u = control[0]
    assert -max_torque <= u <= max_torque, f"Control input u is out of bounds: {u}"

    newthdot = th_dot + (3.0 * g / (2.0 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = np.clip(
        newthdot, -max_speed, max_speed
    )  # This clip is needed or use VectorBound.enforceBounds()
    assert (
        -max_speed <= newthdot <= max_speed
    ), f"New State theta_dot is out of bounds: {newthdot}"

    # This might violate bounds
    newth = th + newthdot * dt
    # newth = ompl_utils.angle_normalize(newth)

    state[0].value = newth
    state[1][0] = newthdot

    # * This part is doing the angle normalization
    SO2 = ob.SO2StateSpace()
    SO2.enforceBounds(state[0])
    assert (
        -pi <= state[0].value <= pi
    ), f"New State theta is out of bounds: {state[0].value}"


def init_rrt(param: Dict[str, Any]):
    # Construct the state space we are planning in
    # *We are planning in [theta theta_dot]

    # Set the SO2 space which in [-max_speed, maxspeed].
    th_space = ob.SO2StateSpace()

    # Set the bounds of omega space to be in [-max_speed, maxspeed].
    omega_space = ob.RealVectorStateSpace(1)
    w_bounds = ob.RealVectorBounds(1)
    w_bounds.setLow(-param["max_speed"])
    w_bounds.setHigh(param["max_speed"])
    omega_space.setBounds(w_bounds)

    # Create compound space which allows the composition of state spaces.
    space = ob.CompoundStateSpace()
    space.addSubspace(th_space, 1.0)
    space.addSubspace(omega_space, 1.0)
    # Lock this state space. This means no further spaces can be added as components.
    space.lock()

    # Create a control space
    cspace = oc.RealVectorControlSpace(space, 1)

    # set the bounds for the control space
    cbounds = ob.RealVectorBounds(1)
    cbounds.setLow(-param["max_torque"])
    cbounds.setHigh(param["max_torque"])
    cspace.setBounds(cbounds)

    # Define a simple setup class
    ss = oc.SimpleSetup(cspace)

    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
    )
    ss.setStatePropagator(oc.StatePropagatorFn(propagate))

    # Create a start state
    start = ob.State(space)
    start[0], start[1] = param["start"][0], param["start"][1]

    # Create a goal state
    goal = ob.State(space)
    goal[0], goal[1] = 0.0, 1.0

    # Set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.05)

    # Creat RRT planner from oc
    si = ss.getSpaceInformation()
    planner = oc.RRT(si)
    # *Set the maximum length of a motion (Only planner from og has this method)
    # planner.setRange(0.1)

    # Set the planner to the SimpleSetup
    ss.setPlanner(planner)
    # Set propagation step size -> duration of each step
    # *(Not using this in propagation_fn, instead uses a constont dt = 0.05)
    si.setPropagationStepSize(0.05)
    # Set optimization objective
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    return ss


def plan(ss: ob.SpaceInformation, param: Dict[str, Any], runtime: float):
    "Attempt to solve the problem" ""
    solved = ss.solve(runtime)
    controlPath = None
    geometricPath = None
    if solved:
        # Print the path to screen
        controlPath = ss.getSolutionPath()
        controlPath.interpolate()
        geometricPath = controlPath.asGeometric()
        geometricPath.interpolate()

        controlPath_np = np.fromstring(
            geometricPath.printAsMatrix(), dtype=float, sep="\n"
        ).reshape(-1, param["state_dim"])
        geometricPath_np = np.fromstring(
            geometricPath.printAsMatrix(), dtype=float, sep="\n"
        ).reshape(-1, param["state_dim"])
        # print("Found solution:\n%s" % path)
    else:
        print("No solution found")
    return controlPath, controlPath_np, geometricPath, geometricPath_np


def plot_th_space(plan_path, render=False, goal=(0, 0)):
    plt.figure()
    plt.plot(plan_path[:, 0], plan_path[:, 1])
    plt.scatter(plan_path[:, 0], plan_path[:, 1])
    plt.scatter(plan_path[0, 0], plan_path[0, 1], c="r", label="start")
    plt.scatter(goal[0], goal[1], c="g", label="goal")
    plt.title("Path in Theta Space")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    # plt.xlim(-np.pi, np.pi)
    # plt.ylim(-8, 8)
    plt.legend()
    if render:
        plt.show()
    else:
        plt.close()


def plot_xy_space(plan_path, render=False):
    plt.figure()
    x_traj = np.sin(plan_path[:, 0])
    y_traj = np.cos(plan_path[:, 0])
    plan_traj = np.asarray([x_traj, y_traj]).T

    x_start, y_start = (x_traj[0], y_traj[0])
    x_goal, y_goal = [0, 1]
    plt.plot(plan_traj[:, 0], plan_traj[:, 1])
    plt.scatter(x=plan_traj[:, 0], y=plan_traj[:, 1])
    plt.scatter(x_start, y_start, c="r", label="start")
    plt.scatter(x_goal, y_goal, c="g", label="goal")
    plt.title("Path in Cartesian Space")
    plt.xlabel(r"x: $\sin{\theta}$")
    plt.ylabel(r"y: $\cos{\theta}$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    if render:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Optimal motion planning with control")
    p.add_argument(
        "--env_id",
        "-env",
        type=str,
        help="Envriment to interact with",
        default="Pendulum-v0",
    )
    p.add_argument(
        "-t",
        "--runtime",
        type=float,
        default=5.0,
        help="(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.",
    )
    p.add_argument(
        "-i",
        "--info",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG. Defaults to WARN.",
    )
    p.add_argument("--seed", help="Random generator seed", type=int, default=0)
    p.add_argument("--plot", "-p", help="Render environment", action="store_true")
    p.add_argument("--render", "-r", help="Render environment", action="store_true")
    p.add_argument("--render_video", "-rv", help="Save a gif", action="store_true")

    args = p.parse_args()

    # Set the OMPL log level
    ompl_utils.setLogLevel(args.info)
    
    # raise overflow / underflow warnings to errors 
    np.seterr(all="raise")

    # Set the random seed
    ompl_utils.setRandomSeed(args.seed)

    # Create the environment
    env = gym.make(args.env_id)
    env.seed(args.seed)
    obs = env.reset()

    # get the state space and state
    cos_th_start, sin_th_start, th_dot_start = obs

    x = sin_th_start
    y = cos_th_start

    # * If both sin and cos are given, then there is a unique solution
    # * theta is computed by arctan2(sin, cos)
    th_start = np.arctan2(x, y)
    assert cos(th_start) == y and sin(th_start) == x

    # Set the parameters of planning
    param = {
        "state_dim": 2,
        "start": [th_start, th_dot_start],
        "goal": [0.0, 0.0],
        "max_theta": pi,
        "max_speed": 8.0,
        "max_torque": 2.0,
        "dt": 0.05,
        "g": 10.0,
        "m": 1.0,
        "l": 1.0,
    }

    ic(param["start"])

    # Plan the path
    ss = init_rrt(param)
    controlPath, controlPath_np, geometricPathm, geometricPath_np = plan(
        ss, param, args.runtime
    )
    ic(geometricPath_np, geometricPath_np.shape)

    # Get info
    # si = ss.getSpaceInformation()
    # pdf = ss.getProblemDefinition()
    # cspace = ss.getControlSpace()
    # state_propagator = ss.getStatePropagator()
    # planner = ss.getPlanner()
    # goal_bias = planner.getGoalBias()

    # Get controls
    controls = controlPath.getControls()
    U = [u[0] for u in controls]

    # Plot the path
    # path = geometricPath_np
    path = controlPath_np
    # * I add negative to theta
    x_traj = -np.sin(path[:, 0])
    y_traj = np.cos(path[:, 0])
    plan_traj = np.asarray([x_traj, y_traj]).T

    if args.plot:
        # Theta vs Theta_dot
        plot_th_space(path, args.plot)

        # x vs y
        plot_xy_space(path, args.plot)
    else:
        plt.close()

    fig = plt.figure()
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    (graph,) = plt.plot([], [], "o-")
    plt.scatter(0, 0, c="r", label="start")
    plt.scatter(0, 1, c="g", label="Goal")
    plt.legend()

    def animate(i):
        if i < plan_traj.shape[0] - 1:
            graph.set_data(plan_traj[:i, 0], plan_traj[:i, 1])

            this_x = [0, plan_traj[i + 1, 0]]
            this_y = [0, plan_traj[i + 1, 1]]
            graph.set_data(this_x, this_y)
        return (graph,)

    ani = FuncAnimation(
        fig, animate, repeat=False, frames=len(plan_traj) + 1, interval=50
    )
    writervideo = animation.FFMpegWriter(fps=60)
    # ani.save('rrt_pendulum_new.gif', writer=writervideo)
    plt.show()

    # Verify the path
    dstate = np.diff(geometricPath_np, axis=0)

    images = []
    for u in U:
        obs, rew, done, info = env.step([u])

        if args.render or args.render_video:
            try:
                if args.render_video:
                    img_array = env.render(mode="rgb_array")
                    img = Image.fromarray(img_array, "RGB")
                    images.append(img)
                else:
                    env.render(mode="human")
                    time.sleep(0.1)
            except KeyboardInterrupt:
                break
    env.close()
    if args.render_video:
        imageio.mimsave("Pendulum_gym.gif", images)
