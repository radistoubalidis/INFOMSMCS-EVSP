from pricing_env import EVSPPricingEnv
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import random
import pulp

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
            distance_km = 0
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
                travel_time = get_deadhead_time(i.end_time_min, dh)
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
    graph = {t: [] for t in trips.trip_number.unique()}

    compat_df = arcs_df[(arcs_df["arc_type"] == "tripDH")]

    for row in compat_df.itertuples(index=False):
        graph[row.from_stop].append((row.to_stop, row.travel_time, row.slack, row.distance_km))

    return graph



# Column Generation Part
def compute_block_cost(block, graph, fixed_cost=1000.0, time_cost_per_min = 0.001):
    if len(block) <= 1:
        return fixed_cost
    
    total_time_cost = 0.0
    for k in range(len(block)-1):
        i_trip, j_trip = block[k], block[k+1]
        
        # Find arc (i_trip → j_trip) in graph[i_trip]
        arc_found = False
        for candidate_trip, travel_time, slack, distance in graph.get(i_trip, []):
            if candidate_trip == j_trip:
                total_time_cost += (travel_time + slack) * time_cost_per_min
                arc_found = True
                break
        
        # If no explicit arc (same-stop connection), assume zero cost
        if not arc_found:
            total_time_cost += 0.0  # or some default cost
    
    return fixed_cost + total_time_cost

def init_columns(trips):
    cols = {}
    for t in trips.trip_number.unique():
        cols[f"col_{len(cols.keys())}"] = {
            "trips": [t],
            "cost": 244.13
        }
    return cols

def solve_master(trips, columns):
    model = pulp.LpProblem("Master", pulp.LpMinimize)
    x = {name: pulp.LpVariable(name, lowBound=0) for name in columns}
    model += pulp.lpSum(columns[name]['cost'] * x[name] for name in columns)
    
    trip_ids = trips.trip_number.unique().tolist()
    named_constraints = {}
    for t in trip_ids:
        constraint = pulp.lpSum(x[name] for name,col in columns.items() if t in col['trips']) == 1
        model += constraint, f"cover_trip_{t}"
        named_constraints[t] = model.constraints[f"cover_trip_{t}"]
    
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    duals = {t: named_constraints[t].pi for t in trip_ids}
    print(f"Dual range: {min(duals.values()):.3f}-{max(duals.values()):.3f}")
    return model, duals




def col_gen_step(trips, graph, columns, arcs_df, policy_net, optimizer, sub_problem='rl', stats=None):
    from agent import rl_pricing
    from pricing_env import greedy_pi_policy
    
    model, duals = solve_master(trips, columns)
    current_objective = pulp.value(model.objective)

    pull_out_trips = arcs_df[arcs_df['arc_type'] == 'pull_out']['to_stop'].tolist()
    pull_out_trips = [
        (row.to_stop, row.travel_time, 0, row.distance_km)  # slack=0 for pull_out
        for row in arcs_df[arcs_df['arc_type'] == 'pull_out'].itertuples()
    ]
    env = EVSPPricingEnv(trips_df=trips, graph=graph, duals=duals, pull_out_trips=pull_out_trips, block_cost=244.13)

    greedy_block, greedy_rc = greedy_pi_policy(env)
    env.reset()

    if policy_net is not None:
        rl_block, rl_rc = rl_pricing(env, policy_net, optimizer)
        if rl_rc < greedy_rc:
            block, reduced_cost = rl_block, rl_rc
            print(f"✅ RL wins: {rl_rc:.3f} vs greedy {greedy_rc:.3f}")
        else:
            block, reduced_cost = greedy_block, greedy_rc
            print(f"Greedy wins: {greedy_rc:.3f} vs RL {rl_rc:.3f}")
    else:
        block, reduced_cost = greedy_block, greedy_rc

    if stats is not None:
        stats['total_episodes'] += 1
        stats['total_columns_generated'] += 1
        if reduced_cost < stats['best_reduced_cost']:
            stats['best_reduced_cost'] = reduced_cost

    print(f"Obj: {current_objective:.2f}, RC: {reduced_cost:.3f}, block len: {len(block)}")

    if reduced_cost >= -1e-3:
        print("No improvement found!")
        return False, model, columns

    block_cost = compute_block_cost(block, graph)
    col_name = f"col_{len(columns)}"
    columns[col_name] = {"trips": block, "cost": block_cost}
    print(f"Greedy RC: {greedy_rc:.3f} | RL RC: {rl_rc:.3f} | Gap: {rl_rc - greedy_rc:.3f}")
    return True, model, columns


