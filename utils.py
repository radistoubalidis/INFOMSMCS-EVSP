from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import random

# def feasible_arcs(trips: pd.DataFrame, deadheads: pd.DataFrame, max_wait = 10):
#     arcs = []

#     for i in trips.itertuples():
#         for j in trips.itertuples():
#             if i.trip_number == j.trip_number:
#                 continue
            
#             if i.to_stop == j.from_stop:
#                 travel_time = 0
#             else:

#                 dh = deadheads[(deadheads.from_stop == i.to_stop) & (deadheads.to_stop == j.from_stop)]

#                 if dh.empty:
#                     continue

#                 travel_time = dh.iloc[0]["time_ver_0"]

#             slack = j.start_time_min - (i.end_time_min + travel_time) # type:ignore

#             if 0 <= slack <= max_wait: # type:ignore
#                 arcs.append((i.trip_number, j.trip_number))

#     return arcs


# build deadhead dictionary
def build_deadhead_dict(deadheads: pd.DataFrame):
    dh_dict = {}

    for row in deadheads.itertuples():
        dh_dict[(row.from_stop, row.to_stop)] = {
            "t0": row.time_ver_0,
            "t1": row.time_ver_1,
            "t2": row.time_ver_2,
            "t3": row.time_ver_3,
            "distance_km": row.distance_km
        }

    return dh_dict

# check in which time interval deadheads fall into
def get_deadhead_time(time_min, dh):
    if  331 <= time_min <= 419 or 541 <= time_min <= 899 or 1141 <= time_min <= 1439 or 1771 <= time_min <= 1859 or 1981 <= time_min <= 1999:
        return dh["t0"]
    elif 420 <= time_min <= 540 or 900 <= time_min <= 1140 or 1860 <= time_min <= 1980:
        return dh["t1"]
    elif 0 <= time_min <= 330 or 1440 <= time_min <= 1770:
        return dh["t2"]
    else:
        return dh["t3"]

# calculate all feasible deadheads
def feasible_arcs(trips: pd.DataFrame, deadhead_dict, depot_stop= "utrgar", max_wait = 240):
    arcs = []
    trips_list = list(trips.itertuples())

    # feasible arcs depot -> trips
    for j in trips_list:
        if depot_stop == j.from_stop:
            travel_time = 0
        else:
            key = (depot_stop, j.from_stop)
            if key not in deadhead_dict:
                continue

            dh = deadhead_dict[key]
            travel_time = dh["t1"] # for the DH from depot to first trip take time with highest value
            distance_km = dh["distance_km"]
        
        arcs.append({
            "arc_type": "pull_out",
            "from_stop": "DEPOT",
            "to_stop": j.trip_number,
            "travel_time": travel_time,
            "distance_km" : distance_km,
            "slack": None
        })

    # feasible arcs trips -> trips
    for i in trips_list:
        for j in trips_list:
            if i.trip_number == j.trip_number:
                continue
            
            if i.to_stop == j.from_stop:
                travel_time = 0
            else:
                key = (i.to_stop, j.from_stop)
                
                if key not in deadhead_dict:
                    continue

                dh = deadhead_dict[key]
                travel_time = get_deadhead_time[i.end_time_min, dh]
                distance_km = dh["distance_km"]

            slack = j.start_time_min - (i.end_time_min + travel_time)

            if 0 <= slack <= max_wait:
                arcs.append({
                    "arc_type": "tripDH",
                    "from_stop": i.trip_number,
                    "to_stop": j.trip_number,
                    "travel_time": travel_time,
                    "distance_km" : distance_km,
                    "slack": slack
                })

    # feasible arcs trips -> depot
    for i in trips_list:
        if i.to_stop == depot_stop:
            travel_time = 0
        else:
            key = (i.to_stop, depot_stop)
            if key not in deadhead_dict:
                continue
            dh_row = deadhead_dict[key]
            travel_time = get_deadhead_time(i.end_time_min, dh_row)
            distance_km = dh["distance_km"]

        arcs.append({
            "arc_type": "pull_in",
            "from_stop": i.trip_number,
            "to_stop": "DEPOT",
            "travel_time": travel_time,
            "distance_km" : distance_km,
            "slack": None
        })
        
    return arcs


class EVSPPricingEnv:
    def __init__(self, trips_df: pd.DataFrame, graph: Dict[int, List[int]], 
                 duals: Dict[int, float], block_cost: float = 1.0, max_len: int = 20):
        """
        graph: {trip_id: [possible_next_trips]} from feasible_arcs()
        duals: {trip_id: π_t} from solve_master()
        """
        self.trips = trips_df.set_index('trip_number')
        self.graph = graph
        self.duals = duals
        self.block_cost = block_cost
        self.max_len = max_len
        
        # All possible trips
        self.all_trips = sorted(list(graph.keys()))
        self.trip_to_idx = {t: i for i, t in enumerate(self.all_trips)}
        self.n_trips = len(self.all_trips)
        
        self.reset()
    
    def reset(self) -> Tuple[int, np.ndarray]:
        """Start new block (before any trip)"""
        self.current_trip = None
        self.block = []         
        self.visited_mask = np.zeros(self.n_trips, dtype=bool)
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> Tuple[int, np.ndarray]:
        """State = (current_trip_idx, visited_mask)"""
        curr_idx = -1 if self.current_trip is None else self.trip_to_idx[self.current_trip]
        return curr_idx, self.visited_mask.copy()
    
    def get_actions(self) -> List[Any]:
        """Possible actions: next trips + STOP"""
        if self.current_trip is None:
            # Can start at any unvisited trip
            available = [t for t in self.all_trips if not self.visited_mask[self.trip_to_idx[t]]]
        else:
            # Successors of current trip
            available = [t for t in self.graph[self.current_trip] 
                        if not self.visited_mask[self.trip_to_idx[t]]]
        
        available.append("STOP")
        return available
    
    def step(self, action: Any) -> Tuple[Tuple[int, np.ndarray], float, bool, Dict]:
        """Take action, return next_state, reward, done, info"""
        self.steps += 1
        reward = 0.0
        done = False
        info = {}
        
        if action == "STOP" or self.steps >= self.max_len:
            # Terminal: compute reduced cost
            sum_pi = sum(self.duals[t] for t in self.block)
            reduced_cost = self.block_cost - sum_pi
            # Reward = -(block_cost - sum(π_t)) = sum(π_t) - block_cost
            reward = -reduced_cost  
            done = True
            info = {
                "block": self.block.copy(),
                "sum_pi": sum_pi,
                "reduced_cost": reduced_cost,
                "block_length": len(self.block)
            }
        else:
            # Add trip to block
            self.current_trip = action
            self.block.append(action)
            trip_idx = self.trip_to_idx[action]
            self.visited_mask[trip_idx] = True
        
        return self._get_state(), reward, done, info


def random_policy(env) -> Tuple[List[int], float]:
    """Random policy for testing (replace later with trained RL)"""
    state = env.reset()
    done = False
    
    while not done:
        actions = env.get_actions()
        action = random.choice(actions)
        state, reward, done, info = env.step(action)
    
    return info["block"], info["reduced_cost"]

def greedy_pi_policy(env) -> Tuple[List[int], float]:
    """Greedy: always pick highest π_t successor"""
    state = env.reset()
    done = False
    
    while not done:
        actions = [a for a in env.get_actions() if a != "STOP"]
        if not actions:
            # No more trips, stop
            _, reward, done, info = env.step("STOP")
            break
            
        # Pick highest π_t
        action = max(actions, key=lambda t: env.duals[t])
        state, reward, done, info = env.step(action)
    
    return info["block"], info["reduced_cost"]


