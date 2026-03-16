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
    deadheads = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, deadheads, depot_stop = "nwggar")
    arcs_df = pd.DataFrame(arcs)
    graph = build_trip_graph_from_arcs_df(trips, arcs_df)
    columns = init_columns(trips)
    
    best_bus_count = float("inf")
    best_cost = float("inf")
    best_solution = None
    best_columns = None

    no_improve_rounds = 0

    step = 5
    max_iters = 20

    print("\nStarting column generation search")

    # Try different CG iteration limits
    for target_iter in range(step, max_iters + step, step):

        print(f"\n=== Running CG with {target_iter} iterations ===")

        # restart from scratch for this target iteration count
        columns = init_columns(trips)

        # run CG
        for i in range(target_iter):
            # one CG step
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

        print(f"Result with {target_iter} CG iterations -> buses={bus_count}, cost={cost:.2f}")

        improved_solution = False

        # minimize number of buses first, then then the costs if number of buses are equal
        if bus_count < best_bus_count:
            improved_solution = True
        elif bus_count == best_bus_count and cost < best_cost:
            improved_solution = True

        # store best solution
        if improved_solution:
            best_bus_count = bus_count
            best_cost = cost
            best_solution = {
                name: var.varValue
                for name, var in final_model.variablesDict().items()
            }
            best_columns = copy.deepcopy(real_columns)
            no_improve_rounds = 0
            print("New best solution found!")
        else:
            no_improve_rounds += 1
            print(f"No improvement round {no_improve_rounds}/3")

        # stop search if no improvement for 3 iterations
        if no_improve_rounds >= 3:
            print("\nStopping search: no improvement for 3 rounds.")
            break

    if best_solution is not None:
        for name, value in best_solution.items():
            if value is not None and value > 0.5:
                print(name, best_columns[name]["trips"])

    print("\nBest solution found:")
    print(f"Total trips: {len(trips)}")
    print(f"Buses used: {best_bus_count}")
    print(f"Total cost: {best_cost:.2f}")
    
            
if __name__ == '__main__':
    main()
