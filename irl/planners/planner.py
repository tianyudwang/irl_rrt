from typing import List, Callable, Tuple

import numpy as np
import torch.multiprocessing as mp 

from irl.planners.sst_planner import SSTPlanner

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
class Planner:
    """Wrapper for the OMPL planners to allow multiprocessing"""
    def __init__(self):
        self.cost_fn = None

    def update_cost(self, cost_fn: Callable) -> None:
        self.cost_fn = cost_fn

    def plan_mp(self, demo_transitions, agent_next_states):
        # Use default os.cpu_count() processes
        with mp.Pool() as pool:
            # Use Pool.starmap to allow multiple arguments
            results = pool.starmap(self.plan_one_expert_state, 
                                   zip(demo_transitions, agent_next_states))
        return zip(*results)

    def plan_one_expert_state(
            self, 
            demo_transition: List[np.ndarray], 
            agent_next_states: List[np.ndarray]
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Given an expert transition (s, a, s') we compute the optimal path from s' 
        to goal. Additionally, for each agent transition (s, a_a, s_a') we compute 
        the optimal path from s_a' to goal.
        """
        planner = SSTPlanner()
        planner.update_ss_cost(self.cost_fn)

        # Given expert transition (s, a, s'), find optimal path from s' to goal
        state, _, _, _, next_state, _ = demo_transition
        demo_path, controls = planner.plan(next_state)

        # Also find agent paths from s_a' to goal
        agent_paths = []
        for next_state in agent_next_states:
            agent_path, controls = planner.plan(next_state)
            agent_paths.append(agent_path)

        # Pad current state to each path
        state = state.reshape(1, -1)
        demo_path = [np.concatenate((state, demo_path), axis=0)]
        agent_paths = [np.concatenate((state, agent_path), axis=0) 
                       for agent_path in agent_paths]
        return demo_path, agent_paths