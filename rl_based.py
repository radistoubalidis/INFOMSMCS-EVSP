from typing import Dict, List, Tuple, Any
from combine_data import parse_dhd, parse_parameters, parse_trips
from utils import feasible_arcs, EVSPPricingEnv, random_policy, greedy_pi_policy, build_deadhead_dict
from utils import feasible_arcs, init_columns, build_trip_graph_from_arcs_df, col_gen_step, build_deadhead_dict
import pprint
import random
import numpy as np
import pandas as pd
import pulp
print("Script started")

# def build_trip_graph(trips: pd.DataFrame, arcs):
#     """
#     arcs: list[(i_trip, j_trip)] from feasible_arcs
#     returns adjacency list: {trip: [successor_trips]}
#     """
#     graph = {t: [] for t in trips.trip_number.unique()}
#     for i, j in arcs:
#         graph[i].append(j)
#     return graph




# Column Generation Part
def init_columns(trips):
    cols = {}
    for t in trips.trip_number:
        cols[f"col_{t}"] = {
            "trips": [t],
            "cost": 1.0
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


def col_gen_step(trips, graph, columns):
    model, duals = solve_master(trips, columns)
    current_objective = pulp.value(model.objective)
    
    env = EVSPPricingEnv(trips, graph, duals)
    block, reduced_cost = random_policy(env)
    
    # print(f"Current obj: {current_objective:.2f}, Found block: {block}, reduced_cost: {reduced_cost:.3f}")
    
    if reduced_cost >= -1e-6:
        print("No improvement found!")
        return False, model, columns
    
    col_name = f"col_{len(columns)}"
    columns[col_name] = {
        "trips": block,
        "cost": 1.0
    }
    # print(f"Added {col_name}: {block}")
    return True, model, columns
    
def solve_final_integer_master(trips, columns):
    model = pulp.LpProblem("FinalMaster", pulp.LpMinimize)
    x = {name: pulp.LpVariable(name, cat="Binary") for name in columns}

    model += pulp.lpSum(columns[name]["cost"] * x[name] for name in columns)

    trip_ids = trips.trip_number.tolist()
    for t in trip_ids:
        model += pulp.lpSum(x[name] for name, col in columns.items() if t in col["trips"]) == 1, f"cover_{t}"

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return model

def main():
    UTR_trips = 'utr/trips.txt'
    trips = parse_trips(UTR_trips)
    UTR_dhd = 'utr/dhd.txt'
    dhd = parse_dhd(UTR_dhd)
    UTR_params = 'utr/parameters.txt'
    params = parse_parameters(UTR_params)
    # trips = pd.read_csv('combined_datasets/trips.csv')
    # dhd = pd.read_csv('combined_datasets/dhd.csv')
    # parameters = pd.read_csv('combined_datasets/parameters.csv')
    deadheads = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, deadheads, depot_stop = "utrgar", max_wait = 1000)
    arcs_df = pd.DataFrame(arcs)
    graph = build_trip_graph_from_arcs_df(trips, arcs_df)
    UTR_trips = 'utr/trips.txt'
    columns = init_columns(trips)
    max_iter=30
    
    for i in range(max_iter):
        improved, model, columns = col_gen_step(trips, graph, columns)
        if not improved:
            break


    print("\nAll generated columns:")

    # for name, col in columns.items():
    #     print(f"{name}: length={len(col['trips'])}, trips={col['trips']}")
    # print("\nFinal columns generated:")

    # for name, var in model.variablesDict().items():
    #     if var.varValue is not None and var.varValue > 1e-6:
    #         print(f"{name}: value={var.varValue:.3f}, trips={columns[name]['trips']}")
    final_model = solve_final_integer_master(trips, columns)
    print("\nFinal integer solution:")
    for name, var in final_model.variablesDict().items():
        if var.varValue > 0.5:
            print(name, columns[name]["trips"])
    print(f"Final Solution: {pulp.value(model.objective):2f} buses")


if __name__ == '__main__':
    main()

