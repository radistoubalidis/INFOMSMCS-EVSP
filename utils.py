from typing import Dict, List, Tuple, Any
import pandas as pd
import itertools
import numpy as np
import random
import pulp

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

def build_trip_graph_from_arcs_df(trips: pd.DataFrame, arcs_df: pd.DataFrame):
    graph = {t: [] for t in trips.trip_number.unique() if t not in [1190,1192,1194]}

    compat_df = arcs_df[arcs_df["arc_type"] == "tripDH"]

    for row in compat_df.itertuples(index=False):
        graph[row.from_stop].append((row.to_stop, row.travel_time, row.slack, row.distance_km))

    return graph



# Column Generation Part
def compute_block_cost(block, graph, fixed_cost=1000.0, time_cost_per_min = 0.1):
    if len(block) <= 1:
        return fixed_cost
    total_time_cost = 0.0
    for k in range(len(block)-1):
        i,j = block[k],block[k+1]
        travel_time, slack = graph[i][j]
        total_time_cost += (travel_time + slack) * time_cost_per_min
    return fixed_cost + total_time_cost

def init_columns(trips):
    cols = {}
    for t in trips.trip_number:
        cols[f"col_{t}"] = {
            "trips": [t],
            "cost": 1000.0
        }
    return cols

def solve_master(trips, columns):
    model = pulp.LpProblem("Master", pulp.LpMinimize)
    x = {name: pulp.LpVariable(name, lowBound=0) for name in columns}
    
    model += pulp.lpSum(columns[name]['cost'] * x[name] for name in columns)
    
    trip_ids = trips.trip_number.tolist()
    for t in trip_ids:
        model += pulp.lpSum(x[name] for name,col in columns.items() if t in col['trips']) == 1, f"cover_{t}"
    
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    duals = {t: model.constraints[f"cover_{t}"].pi for t in trip_ids}
    return model, duals


def col_gen_step(trips, graph, columns, sub_problem='rl'):
    model, duals = solve_master(trips, columns)
    current_objective = pulp.value(model.objective)
    
    if sub_problem == 'rl':
        env = EVSPPricingEnv(trips_df=trips, graph=graph, duals=duals)
        block, reduced_cost = greedy_pi_policy(env)
    elif sub_problem == 'metaheuristics':
        pass
    
    print(f"Current obj: {current_objective:.2f}, Found block: {block}, reduced_cost: {reduced_cost:.3f}")
    
    if reduced_cost >= -1e-3:
        print("No improvement found!")
        return False, model, columns
    
    block_cost = compute_block_cost(block, graph)
    col_name = f"col_{len(columns)}"
    columns[col_name] = {
        "trips": block,
        "cost": block_cost
    }
    print(f"Added {col_name}: (cost={block:.1f})")
    return True, model, columns

class EVSPPricingEnv:
    def __init__(self, trips_df: pd.DataFrame, graph: Dict[int, List[int]], 
                 duals: Dict[int, float], block_cost: float = 1000.0, time_cost_per_min: float = 0.1, max_len: int = 30):
        self.trips = trips_df.set_index('trip_number')
        self.graph = graph # {i_trip: (j_trip, travel_time, slack)}
        self.duals = duals
        self.block_cost = block_cost
        self.time_cost_per_min = time_cost_per_min
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
        self.cumulative_pi = 0.0
        self.cumulative_time_cost = 0.0
        return self._get_state()
    
    def _get_state(self) -> Tuple[int, np.ndarray]:
        curr_idx = -1 if self.current_trip is None else self.trip_to_idx[self.current_trip[0]]
        return (curr_idx, self.visited_mask.copy(), self.cumulative_pi, self.cumulative_time_cost)
    
    def get_actions(self) -> List[Any]:
        if self.current_trip is None:
           return list(itertools.chain.from_iterable(self.graph.values()))
        else:
           return list(self.graph[self.current_trip[0]])
            
    def step(self, action: Any):
        self.steps += 1
        reward = 0.0
        done = False
        info = {}
        
        if action == "STOP" or self.steps >= self.max_len:
            # Terminal: compute reduced cost
            reduced_cost = (self.block_cost + self.cumulative_time_cost) - self.cumulative_pi
            reward = -reduced_cost  
            done = True
            info = {
                "block": self.block.copy(),
                "sum_pi": self.cumulative_pi,
                'total_time_cost': self.cumulative_time_cost,
                "reduced_cost": reduced_cost,
                "block_length": len(self.block)
            }
        else:
            # add arc cost from previous trip
            if self.current_trip is not None:
                travel_time, slack, distance = action[1], action[2], action[3]
                arc_cost = (travel_time + slack) * self.time_cost_per_min
                self.cumulative_time_cost += arc_cost
            
        self.current_trip = action
        self.block.append(action)
        self.cumulative_pi += self.duals[action[0]]
        
        reward = self.duals[action[0]] - (self.cumulative_time_cost / len(self.block))
        if action in self.trip_to_idx:
            trip_idx = self.trip_to_idx[action[0]]
            self.visited_mask[trip_idx] = True
        
        return self._get_state(), reward, done, info


def random_policy(env: EVSPPricingEnv) -> Tuple[List[int], float]:
    """Random policy for testing (replace later with trained RL)"""
    # state = env.reset()
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
            state, _, done, info = env.step("STOP")
            break
        
        if env.current_trip is None:
            action = max(actions, key=lambda t: env.duals[t])  
        else:
            action = max(actions, key=lambda j: env.duals[j] - env.graph[env.current_trip][j][0] * env.time_cost_per_min)
    
        state, _, done, info = env.step(action)
    
    return info['block'], info['reduced_cost']