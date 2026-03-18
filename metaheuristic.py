from combine_data import parse_dhd, parse_parameters, parse_trips
from utils import feasible_arcs, build_deadhead_dict,init_columns, build_trip_graph_from_arcs_df, col_gen_step, solve_final_integer_master, rebuild_columns_with_real_costs
import pandas as pd
import pulp
import copy


def main():
    UTR_trips = 'utr/trips.txt'
    trips = parse_trips(UTR_trips)
    UTR_dhd = 'utr/dhd.txt'
    dhd = parse_dhd(UTR_dhd)
    UTR_params = 'utr/parameters.txt'
    params = parse_parameters(UTR_params)

    # trips = pd.read_csv('combined_datasets/trips.csv')
    # trips['trip_number'] = trips['line_number'].astype(str) + '_' + trips['trip_number'].astype(str)
    # dhd = pd.read_csv('combined_datasets/dhd.csv')

    deadheads = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, deadheads, depot_stop = "nwggar")
    arcs_df = pd.DataFrame(arcs)
    graph = build_trip_graph_from_arcs_df(trips, arcs_df)
    columns = init_columns(trips)

    max_iters = 20

    print(f"\nStarting column generation with {max_iters} iterations")

    for i in range(max_iters):
 
        improved, model, columns = col_gen_step(trips=trips, graph=graph, columns=columns, sub_problem="metaheuristics")

        if not improved:
            print("No improving column found.")
            break

    print(f"Total columns generated: {len(columns)}")


    # rebuild column costs using real cost values
    real_columns = rebuild_columns_with_real_costs(
        trips=trips,
        graph=graph,
        columns=columns,
        real_fixed_cost=244.13,
        cost_per_km=0.13
    )

    # solve the problem using binary variables
    final_model = solve_final_integer_master(trips, real_columns)

    # count number of buses used in solution
    bus_count = sum(
        1 for var in final_model.variablesDict().values()
        if var.varValue is not None and var.varValue > 0.5
    )

    # compute total cost
    cost = pulp.value(final_model.objective)

    for name, var in final_model.variablesDict().items():
        if var.varValue is not None and var.varValue > 0.5:
            print(name, real_columns[name]["trips"])

    print("\nBest solution found:")
    print(f"Total trips: {len(trips)}")
    print(f"Buses used: {bus_count}")
    print(f"Total cost: {cost:.2f}")
    
            
if __name__ == '__main__':
    main()
