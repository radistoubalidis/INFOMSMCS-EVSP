from combine_data import parse_dhd, parse_parameters, parse_trips
from utils import (
    feasible_arcs,
    build_deadhead_dict,
    init_columns,
    build_trip_graph_from_arcs_df,
    col_gen_step,
)
from rl_based import solve_final_integer_master
import pandas as pd
import pulp
print("Script started")

def solve_final_integer_master(trips, columns):
    model = pulp.LpProblem("FinalIntegerMaster", pulp.LpMinimize)

    x = {
        name: pulp.LpVariable(name, lowBound=0, upBound=1, cat="Binary")
        for name in columns
    }

    model += pulp.lpSum(columns[name]["cost"] * x[name] for name in columns)

    trip_ids = trips.trip_number.tolist()
    for t in trip_ids:
        model += pulp.lpSum(
            x[name] for name, col in columns.items() if t in col["trips"]
        ) == 1, f"cover_{t}"

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return model

def build_trip_km_lookup(trips: pd.DataFrame):
    return {
        row.trip_number: row.distance_km
        for row in trips.itertuples(index=False)
    }

def main():
    UTR_trips = 'utr/trips.txt'
    trips = parse_trips(UTR_trips)
    UTR_dhd = 'utr/dhd.txt'
    dhd = parse_dhd(UTR_dhd)
    UTR_params = 'utr/parameters.txt'
    params = parse_parameters(UTR_params)
    deadheads = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, deadheads, depot_stop = "utrgar", max_wait = 1000)
    arcs_df = pd.DataFrame(arcs)
    graph = build_trip_graph_from_arcs_df(trips, arcs_df)
    columns = init_columns(trips)
    max_iter = 10

    for i in range(max_iter):
        print(f"\n--- Column generation iteration {i+1} ---")
        improved, model, columns = col_gen_step(
            trips=trips,
            graph=graph,
            columns=columns,
            sub_problem="metaheuristics"
        )
        if not improved:
            break

    final_model = solve_final_integer_master(trips, columns)

    print("\nFinished column generation.")
    print(f"Final integer objective = {pulp.value(final_model.objective):.2f}")

    selected_columns = []
    for name, var in final_model.variablesDict().items():
        if var.varValue is not None and var.varValue > 0.5:
            selected_columns.append((name, columns[name]["trips"], columns[name]["cost"]))

    print("\nFinal columns used:")
    for name, trips_in_col, cost in selected_columns:
        print(f"{name}: cost={cost:.2f}, trips={trips_in_col}")

    num_buses = len(selected_columns)
    print(f"\nNumber of buses used: {num_buses}")


    non_singletons = [col for col in columns.values() if len(col["trips"]) > 1]
    print("Total columns:", len(columns))
    print("Non-singleton columns:", len(non_singletons))
    print("Singleton columns:", sum(1 for col in columns.values() if len(col["trips"]) == 1))
    selected_singletons = 0
    selected_non_singletons = 0

    for name, var in final_model.variablesDict().items():
        if var.varValue > 0.5:
            if len(columns[name]["trips"]) == 1:
                selected_singletons += 1
            else:
                selected_non_singletons += 1

    print("Selected singleton columns:", selected_singletons)
    print("Selected non-singleton columns:", selected_non_singletons)   
if __name__ == '__main__':
    main()
