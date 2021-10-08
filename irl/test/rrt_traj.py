import argparse
import sys
import os
import time
from typing import Any, Dict
import random
from math import sin, cos, pi

from PIL import Image
import imageio

import gym
import numpy as np
import matplotlib.pyplot as plt

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc
from ompl import geometric as og


from icecream import ic


class StateValidityChecker(ob.StateValidityChecker):
    """
    State Validation Check
    Returns whether the given state's position overlaps the obstacle.
    """
    def __init__(self, si, l=1):
        super().__init__(si)
        self.l = l
        self.si = si    
    
    def isValid(self, state) -> bool:
        """
        For a single pandulum the trajectory should be a curve
        """
        # TODO: 2d or ompl's method
        valid = self.cartesianSpaceValid(state) and self.si.satisfiesBounds(state)
        
        #  ! change this back
        # return valid
        return True
        
    def cartesianSpaceValid(self, state) -> bool:
        """Dummy 2D cartersian space Check"""
        # length of the pendulum
        theta = state[0]
        x, y = np.sin(theta), np.cos(theta)
        return np.hypot(x, y) == self.l


# class KinemticPendulumODE(ob.realVectorStateSpace):
    
#     def __init__(self, space):
#         super().__init__(space)
#         self.space = space
#         self.dim = space.getDimension()
#         self.g = 10.0
#         self.m = 1.0
#         self.l = 1.0
#         self.dt = 0.05
#         self.max_speed = 8.0
#         self.max_torque = 2.0
    
#     def operator(self, state, control, dstate) -> None:
#         """
#         The state is a vector of theta and theta_dot
#         """
#         th, th_dot = state[0], state[1]
#         u = control[0]   # may cause error
#         duration = self.dt
        
#         u = np.clip(u, -self.max_torque, self.max_torque) # ? Do we need this?
#         newthdot = th_dot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * control) * duration
#         newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
#         newth = th + newthdot * duration
        
#         dstate[0] = newth
#         dstate[1] = newthdot
    
#     def update(self, state, dstate):
#         """
#         Update the state
#         """
#         state[0] = dstate[0]
#         state[1] = dstate[1]

def propagate(start: ob.State, control: oc.Control, duration: float, state: ob.State):
    
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    max_speed = 8.0
    max_torque = 2.0
    
    t = dt
    ic("here")
    while t + sys.float_info.min < duration:
        th, th_dot = state[0], state[1]
        u = control[0]   # may cause error

        u = np.clip(u, -max_torque, max_torque) # ? Do we need this?
        newthdot = th_dot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * control) *  dt
        newthdot = np.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * duration
        state[0] = newth
        start[1] = newthdot
        t += dt
    
    if t + sys.float_info.min > duration:
        th, th_dot = state[0], state[1]
        u = control[0]   # may cause error

        u = np.clip(u, -max_torque, max_torque) 
        newthdot = th_dot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * control) *  dt
        newthdot = np.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * duration
        state[0] = newth
        start[1] = newthdot
        t += dt
    
     
def init_rrt(param: Dict[str, Any]):
    # Construct the robot state space in which we're planning.
    # *We are planning in [thetam theta_dot]
    space = ob.RealVectorStateSpace(2)

    # Set the bounds of space to be in [-pi,pi] x [-max_speed, maxspeed].    
    bounds = ob.RealVectorBounds(2)
    bounds.low[0] =  -param["max_theta"]
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
    
    # create a simple setup object
    ss = oc.SimpleSetup(cspace)
    
    # Construct a space information instance for this state space
    si = ss.getSpaceInformation()
    # Set the object used to check which states in the space are valid
    ss.setStateValidityChecker(StateValidityChecker(si))
    # Set the object used to check which motion betwwen states are valid  # ! not Implemented
    # motion_valid_checker = MotionValidator(si, boundary, blocks)
    # si.setMotionValidator(motion_valid_checker)
    
    # Set State Propagator
    ss.setStatePropagator(oc.StatePropagatorFn(propagate))    
    # Optimization Objective
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    
    
    # Set start and goal position
    start = ob.State(space)
    start[0], start[1] = param["start"][0], param["start"][1]
    
    
    goal = ob.State(space)
    goal[0], goal[1] = param["goal"][0], param["goal"][1]
    ss.setStartAndGoalStates(start, goal)
    
    # Init up RRT* planner
    planner = og.RRTstar(si)
    # Set the maximum length of a motion
    planner.setRange(0.05)  # TODO: This is the funtion change step size 
    
    # Set up RRT* planner
    ss.setPlanner(planner)
    # (optionally) set propagation step size
    si.setPropagationStepSize(0.05)
    
    return ss

def rrt_plan(
    param: Dict[str, Any],
    ss: og.SimpleSetup,
    runTime: float,
):
    """Attempt to solve the planning problem in the given runtime"""
    solved = ss.solve(runTime)
    path = None

    if solved:
        # Output the length of the path found
        path = ss.getSolutionPath().printAsMatrix()
        path = np.fromstring(path, dtype=float, sep='\n').reshape(-1, param["state_dim"])
    else:
        print("No solution found.")
        
    return path
    

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
        '-t', '--runtime', type=float, default=2.0,
        help='(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.'
    )
    p.add_argument(
        '-i', '--info', type=int, default=0, choices=[0, 1, 2],
        help='(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG. Defaults to WARN.'
    )
    p.add_argument("--seed", help="Random generator seed", type=int, default=42)
    p.add_argument("--render", '-r', help="Render environment", action="store_true")
    p.add_argument("--render_video", '-rv', help="Save a gif", action="store_true")
    
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
    
    
    np.seterr(all='raise')
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create the environment    
    env = gym.make(args.env_id)
    env.seed(args.seed)
    obs = env.reset()
    
    # get the state space and state
    cos_th_start, sin_th_start, th_dot_start = obs
    
    # * If both sin and cos are given, then there is a unique solution
    # * theta is computed by arctan2(sin, cos)
    
    th_start = np.arctan2(sin_th_start, cos_th_start)
    assert cos(th_start) == cos_th_start and sin(th_start) == sin_th_start    
    
    # Set the parameters of planning    
    param = {
        "state_dim": 2,
        "start": [th_start, th_dot_start],
        "goal": [0 , 0],
        "max_theta": pi,
        "max_speed": 8.0,
        "max_torque": 2.0,
        "dt": 0.05,
        "g": 10.0,
        "m": 1.0,
        "l": 1.0,
    }

    ss = init_rrt(param)
    path = rrt_plan(param, ss, args.runtime)
    
    # Theta vs Theta_dot
    plt.figure()
    plt.plot(path[:,0], path[:,1])
    plt.scatter(path[0,0], path[0,1], c="r", label="start")
    plt.scatter(path[-1,0], path[-1,1], c="g", label="goal")
    plt.title("Path in Theta Space")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    # plt.xlim(-np.pi, np.pi)
    # plt.ylim(-8, 8)
    plt.legend()
    plt.show()
    
    # x vs y
    x_traj = np.sin(path[:,0])
    y_traj = np.cos(path[:,0])
    plan_traj = np.asarray([x_traj, y_traj]).T
    
    x_start, y_start = (sin_th_start, cos_th_start)
    x_goal, y_goal = [0, 1]
    plt.scatter(x=plan_traj[:, 0], y=plan_traj[:, 1])
    plt.scatter(x_start, y_start, c="r", label="start")
    plt.scatter(x_goal, y_goal, c="g", label="goal")
    plt.title("Path in Cartesian Space")
    plt.xlabel(r"x: $\sin{\theta}$")
    plt.ylabel(r"y: $\cos{\theta}$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [plt.close() if event.key in ["escape"] else None],
    )
    plt.show()    

    true_taj = [[x_start, y_start]]
    images = []
    step = 0
    for th, th_dot in path:
        obs, act, rew, info = env.step([th, th_dot])
        cos_th_yt, sin_th_xt, th_dot_t = obs
        true_taj.append([sin_th_xt, cos_th_yt])
        if args.render:
            env.render(mode='rgb_array')
            time.sleep(0.5)
            step += 1
            print(f"Step: {step}", end="\r")
        elif args.render_video:
            img_array = env.render(mode='rgb_array')
            img = Image.fromarray(img_array, 'RGB')
            img = img.resize((500, 500))
            images.append(img)
        
    video_path = os.path.join(os.getcwd(), "video.gif")
    if args.render_video:
        imageio.mimsave(video_path, images)
    
    true_taj = np.asarray(true_taj)
    # ic(plan_traj, true_taj)
    plt.scatter(true_taj[:,0], true_taj[:,1])
    plt.title("True Trajectory from the environment")
    plt.show()