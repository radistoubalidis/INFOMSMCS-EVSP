from typing import Dict, List, Tuple, Any
from combine_data import parse_dhd, parse_parameters, parse_trips
from utils import (
    feasible_arcs, 
    EVSPPricingEnv, 
    build_deadhead_dict, 
    init_columns, 
    build_trip_graph_from_arcs_df, 
    col_gen_step
)
import pprint
import random
import numpy as np
import pandas as pd
import pulp
print("Script started")
    
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
    # params = pd.read_csv('combined_datasets/parameters.csv')
    deadheads = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, deadheads, depot_stop = "utrgar", max_wait = 1000)
    arcs_df = pd.DataFrame(arcs)
    graph = build_trip_graph_from_arcs_df(trips, arcs_df)
    UTR_trips = 'utr/trips.txt'
    columns = init_columns(trips)
    max_iter = 10000
    
    for i in range(max_iter):
        improved, model, columns = col_gen_step(trips, graph, columns)
        if not improved:
            break


    print("\nAll generated columns:")
    final_model = solve_final_integer_master(trips, columns)
    print("\nFinal integer solution:")
    for name, var in final_model.variablesDict().items():
        if var.varValue > 0.5:
            print(name, columns[name]["trips"])
    print(f"Final Solution: {pulp.value(model.objective):2f} buses")


if __name__ == '__main__':
    main()

