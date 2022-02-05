from dataclasses import dataclass
from typing import List
import numpy as np
import torch as th

@dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""
    
    states: np.ndarray
    """States, shape (trajectory_len + 1, ) + state_shape."""

    actions: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    def __len__(self):
        """Returns number of transitions, equal to the number of actions."""
        return len(self.actions)

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.states) != len(self.actions) + 1:
            raise ValueError(
                "expected one more observations than actions: "
                f"{len(self.states)} != {len(self.actions)} + 1",
            )
        if len(self.actions) == 0:
            raise ValueError("Degenerate trajectory: must have at least one action.")

@dataclass(frozen=True)
class TrajectoryWithReward(Trajectory):
    """A `Trajectory` that additionally includes reward information."""

    rewards: np.ndarray
    """Reward, shape (trajectory_len, ). dtype float."""

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()

        if self.rewards.shape != (len(self.actions),):
            raise ValueError(
                "rewards must be 1D array, one entry for each action: "
                f"{self.rewards.shape} != ({len(self.actions)},)",
            )


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    """State, shape (observation_shape, )."""

    action: np.ndarray
    """Action, shape (action_shape, )."""

    next_state: np.ndarray
    """Next state, shape (observation_shape, )."""

    def __len__(self):
        """Length of a transition is always 1"""
        return 1

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.state.shape) > 1 or len(self.action.shape) > 1 or len(self.next_state.shape) > 1:
            raise ValueError(
                "Initialiazed more than one transition"
            )
        if self.state.shape != self.next_state.shape:
            raise ValueError(
                "state and next state have different dimensions in one transition"
            )


def convert_trajectories_to_transitions(trajectories: List[Trajectory]) -> List[Transition]:
    """Flatten a series of trajectories to a series of transitions"""
    assert len(trajectories) >= 1, "Cannot convert empty trajectory"

    transitions = []
    for traj in trajectories:
        for i in range(len(traj)):
            transition = Transition(traj.states[i], traj.actions[i], traj.states[i+1])
            transitions.append(transition)

    assert len(transitions) == sum([len(traj) for traj in trajectories]), (
        "Number of transitions does not match after conversion"
    )
    return transitions