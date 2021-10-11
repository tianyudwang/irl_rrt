import argparse
import sys
import os
import time
from typing import Any, Dict
import random

from math import sin, cos, pi
from functools import partial

import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


global steps
steps = 0

def angle_normalize(x) -> float:
    """"""
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def isStateValid(si: ob.SpaceInformation, state: ob.State) -> bool:
    # perform collision checking or check if other constraints are satisfied
    l = 1
    th, th_dot = state[0], state[1]
    x, y = np.sin(th), np.cos(th)
    
    # position valid
    pos_inbound = np.hypot(x, y) == l

    return si.satisfiesBounds(state) and pos_inbound 


def propagate(
    start: ob.State,
    control: oc.Control,
    duration: float,
    state: ob.State
):
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    max_speed = 8.0
    max_torque = 2.0

    th, th_dot = start[0], start[1]
    assert -pi <= th <= pi, f"State theta is out of bounds: {th}"
    assert -max_speed <= th_dot <= max_speed, f"State theta_dot is out of bounds: {th_dot}" 
    
    u = control[0]
    assert -max_torque <= u <= max_torque, f"Control input u is out of bounds: {u}"
    
    newthdot = th_dot + (3.0 * g / (2.0 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = np.clip(newthdot, -max_speed, max_speed)
    assert -max_speed <= newthdot <= max_speed, f"New State theta_dot is out of bounds: {newthdot}"
    
    newth = th + newthdot * dt
    newth = angle_normalize(newth)
    assert -pi <= newth <= pi, f"New State theta is out of bounds: {newth}"
    
    
    state[0] = newth
    state[1] = newthdot
    assert state[0] == newth and state[1] == newthdot
    
    # print("here")


def plan(param: Dict[str, Any], runtime: float):
    # construct the state space we are planning in
    # *We are planning in [theta theta_dot]

    space = ob.RealVectorStateSpace(2)

    # Set the bounds of space to be in [-pi,pi] x [-max_speed, maxspeed].
    bounds = ob.RealVectorBounds(2)
    bounds.low[0] = -param["max_theta"]
    bounds.high[0] = param["max_theta"]
    bounds.low[1] = -param["max_speed"]
    bounds.high[1] = param["max_speed"]
    space.setBounds(bounds)

    # create a control space
    cspace = oc.RealVectorControlSpace(space, 1)

    # set the bounds for the control space
    cbounds = ob.RealVectorBounds(1)
    cbounds.setLow(-param["max_torque"])
    cbounds.setHigh(param["max_torque"])
    cspace.setBounds(cbounds)

    # define a simple setup class
    ss = oc.SimpleSetup(cspace)

    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
    )
    ss.setStatePropagator(oc.StatePropagatorFn(propagate))

    # create a start state
    start = ob.State(space)
    start[0], start[1] = param["start"][0], param["start"][1]

    # create a goal state
    goal = ob.State(space)
    goal[0], goal[1] = param["goal"][0], param["goal"][1]

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal)

    # (optionally) set planner
    si = ss.getSpaceInformation()
    planner = oc.RRT(si)
    # *Set the maximum length of a motion (Only planner from og has this method)
    # planner.setRange(0.1)  
    planner.setGoalBias(0.05) # Deafult is 0.05
    
    ss.setPlanner(planner)
    # (optionally) set propagation step size
    si.setPropagationStepSize(0.05)
    # ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    # attempt to solve the problem
    solved = ss.solve(runtime)
    path = None
    if solved:
        # print the path to screen
        path = ss.getSolutionPath().printAsMatrix()
        path = np.fromstring(path, dtype=float, sep="\n").reshape(
            -1, param["state_dim"]
        )
        # print("Found solution:\n%s" % path)
    else:
        print("No solution found")
    return path, ss


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

def recover_control(path):
    
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    max_speed = 8.0
    max_torque = 2.0

    control = []
    
    for i in range(1, path.shape[0] - 1):
        th, th_dot = path[i-1, 0], path[i-1, 1]
        newth, newthdot = path[i, 0], path[i, 1]
        a = (newthdot - th_dot) / dt 
        u = a - (3 * g / (2 * l) * np.sin(th)) / 3.0 * (m * l ** 2)
        assert -max_torque <= u <= max_torque, f"Control input u is out of bounds: {u}"
        control.append(u)    
    return control

if __name__ == "__main__":
    p = argparse.ArgumentParser()
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

    # Set the log level
    if args.info == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif args.info == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif args.info == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")

    np.seterr(all="raise")
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create the environment
    env = gym.make(args.env_id)
    env.seed(args.seed)
    obs = env.reset()

    # get the state space and state
    cos_th_start, sin_th_start, th_dot_start = obs

    # * sin_th need to add negative
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

    # Plan the path
    path, ss = plan(param, args.runtime)
    ic(path, path.shape)

    
    si = ss.getSpaceInformation()
    pdf = ss.getProblemDefinition()
    cspace = ss.getControlSpace () 
    state_propagator = ss.getStatePropagator()
    planner = ss.getPlanner ()
    goal_bias = planner.getGoalBias ()
    
    
    

    # Plot the path
    x_traj = np.sin(path[:, 0])
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
    plt.scatter(0, 0, c='r', label="start")
    plt.scatter(0, 1, c='g', label='Goal')
    plt.legend()

    def animate(i):
        if i < plan_traj.shape[0] - 1:
            graph.set_data(plan_traj[:i, 0], plan_traj[:i, 1])
            
            this_x = [0, plan_traj[i+1, 0]]
            this_y = [0, plan_traj[i+1, 1]]
            graph.set_data(this_x, this_y)
        return graph,

    ani = FuncAnimation(fig, animate, repeat=False, frames=len(plan_traj)+1, interval=500,)
    # ani.save('rrt_pendulum.gif', writer='imagemagick', fps=3)
    plt.show()

    # Verify the path
    dstate = np.diff(path, axis=0)
    
    # for dX in dstate:
        # distance = np.linalg.norm(dX)
        
    recover_control(path)