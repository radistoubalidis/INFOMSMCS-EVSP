import pandas as pd
import numpy as np
import itertools
import random
from typing import Dict, List, Tuple, Any

class EVSPPricingEnv:
    def __init__(self, trips_df: pd.DataFrame, graph: Dict[int, List[int]],duals: Dict[int, float],
                 pull_out_trips: List[int],
                 block_cost: float = 244.13, time_cost_per_min: float = 0.1, max_len: int = 30
    ):
        self.trips = trips_df.set_index('trip_number')
        self.graph = graph
        self.duals = duals
        self.pull_out_trips = pull_out_trips
        self.block_cost = block_cost
        self.time_cost_per_min = time_cost_per_min
        self.max_len = max_len
        self.all_trips = sorted(list(graph.keys()))
        self.trip_to_idx = {t: i for i, t in enumerate(self.all_trips)}
        self.n_trips = len(self.all_trips)
        self.reset()

    def reset(self):
        self.current_trip = None
        self.block = []
        self.visited_mask = np.zeros(self.n_trips, dtype=bool)
        self.steps = 0
        self.cumulative_pi = 0.0
        self.cumulative_time_cost = 0.0
        return self._get_state()

    def _get_state(self):
        curr_idx = -1 if self.current_trip is None else self.trip_to_idx[self.current_trip[0]]
        return (curr_idx, self.visited_mask.copy(), self.cumulative_pi, self.cumulative_time_cost)

    def get_actions(self):
        if self.current_trip is None:
            return self.pull_out_trips
        else:
            return list(self.graph[self.current_trip[0]])

    def step(self, action: Any):
        self.steps += 1
        done = False
        info = {}

        if action == "STOP" or self.steps >= self.max_len:
            reduced_cost = (self.block_cost + self.cumulative_time_cost) - self.cumulative_pi
            reward = -reduced_cost
            done = True
            info = {
                "block": self.block.copy(),
                "sum_pi": self.cumulative_pi,
                "total_time_cost": self.cumulative_time_cost,
                "reduced_cost": reduced_cost,
                "block_length": len(self.block)
            }
            return self._get_state(), reward, done, info

        if isinstance(action, tuple):
            trip_num = action[0]
            travel_time, slack = action[1], action[2]
        else:
            trip_num = action
            travel_time, slack = 0, 0

        if self.current_trip is not None:
            arc_cost = (travel_time + slack) * self.time_cost_per_min
            self.cumulative_time_cost += arc_cost

        self.current_trip = (trip_num,)
        self.block.append(trip_num)
        self.cumulative_pi += self.duals[trip_num]

        reward = 0.0

        if trip_num in self.trip_to_idx:
            self.visited_mask[self.trip_to_idx[trip_num]] = True

        return self._get_state(), reward, done, info


def random_policy(env):
    done = False
    while not done:
        actions = env.get_actions()
        if not actions:
            _, _, done, info = env.step("STOP")
            break
        action = random.choice(actions)
        _, _, done, info = env.step(action)
    return info["block"], info["reduced_cost"]


def score_action(env, action):
    return env.duals[action[0]] \
        - env.time_cost_per_min * action[1] \
        - env.time_cost_per_min * action[2]


def greedy_pi_policy(env):
    env.reset()
    done = False
    while not done:
        actions = [a for a in env.get_actions() if a != "STOP"]
        if not actions:
            _, _, done, info = env.step("STOP")
            break
        if env.current_trip is None:
            action = max(actions, key=lambda t: env.duals[t[0]])
        else:
            action = max(actions, key=lambda t: score_action(env, t))
        _, _, done, info = env.step(action)
    return info['block'], info['reduced_cost']
