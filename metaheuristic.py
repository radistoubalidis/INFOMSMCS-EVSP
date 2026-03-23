from combine_data import parse_dhd, parse_parameters, parse_trips
from utils import feasible_arcs, build_deadhead_dict,init_columns, build_trip_graph_from_arcs_df, col_gen_step, solve_final_integer_master, BusParams
import pandas as pd
import pulp
from time import perf_counter
import datetime
import json 
import os

def build_depot_energy_lookup(arcs_df, energy_consumption_per_km):
    pull_out_energy = {}
    pull_in_energy = {}

    for row in arcs_df.itertuples(index=False):

        if row.arc_type == "pull_out":
            # DEPOT -> trip
            pull_out_energy[row.to_stop] = row.distance_km * energy_consumption_per_km

        elif row.arc_type == "pull_in":
            # trip -> DEPOT
            pull_in_energy[row.from_stop] = row.distance_km * energy_consumption_per_km

    return pull_out_energy, pull_in_energy

def compute_solution_times(final_model, columns, graph, trip_time):
    total_trip_time = 0.0
    total_deadhead_time = 0.0
    total_waiting_time = 0.0

    for name, var in final_model.variablesDict().items():
        if var.varValue is None or var.varValue <= 0.5:
            continue

        block = columns[name]["trips"]

        # trip time
        for t in block:
            total_trip_time += trip_time[t]

        # deadhead + waiting
        for k in range(len(block) - 1):
            i = block[k]
            j = block[k + 1]

            arc = next((a for a in graph.get(i, []) if a[0] == j), None)
            if arc:
                travel_time = arc[1]
                slack = arc[2]

                total_deadhead_time += travel_time
                total_waiting_time += slack

    total_operational_time = (
        total_trip_time + total_deadhead_time + total_waiting_time
    )

    return {
        "trip_time": total_trip_time,
        "deadhead_time": total_deadhead_time,
        "waiting_time": total_waiting_time,
        "total_time": total_operational_time,
    }

def save_viz_data(buses: list[float], costs: list[float], total_cost: float, dataset, max_iters, total_buses:int , runtime: float):
    os.makedirs("viz_data", exist_ok=True)

    viz_data_path = f'viz_data/metaheuristic_{dataset}_iters{max_iters}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    
    viz_data = {'steps': [], 'total_cost': total_cost, 'total_buses': total_buses, 'runtime': runtime}
    
    for i in range(len(buses)):
        viz_data['steps'].append({
            'step': i,
            'buses': buses[i],
            'cost': costs[i]
        })
        
    with open(viz_data_path, 'w') as f:
        json.dump(viz_data, f, indent=4)

    print(f"Saved visualization data to {viz_data_path}")

def main():
    start = perf_counter()

    dataset = "qlink_3"

    UTR_trips = f'{dataset}/trips.txt'
    trips = parse_trips(UTR_trips)
    UTR_dhd = f'{dataset}/dhd.txt'
    dhd = parse_dhd(UTR_dhd)
    UTR_params = f'{dataset}/parameters.txt'
    params = parse_parameters(UTR_params)
    bus_params = BusParams.from_params_df(params)

    deadheads = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, deadheads, depot_stop = "nwggar")
    arcs_df = pd.DataFrame(arcs)
    pull_out_energy, pull_in_energy = build_depot_energy_lookup(arcs_df, bus_params.energy_per_km)
    graph = build_trip_graph_from_arcs_df(trips, arcs_df)
    columns = init_columns(trips, bus_params.fixed_cost)


    max_iters = 1000

    print(f"\nStarting column generation with {max_iters} iterations")

    buses_history = []
    costs_history = []

    total_destroy_counts = {}
    total_repair_counts = {}

    for i in range(max_iters):
 
        improved, model, columns, destroy_count, repair_count = col_gen_step(trips=trips, graph=graph, columns=columns,
        pull_out_energy=pull_out_energy,
        pull_in_energy=pull_in_energy,
        bus_params=bus_params
        )

        # accumulate destroy counts
        for k, v in destroy_count.items():
            total_destroy_counts[k] = total_destroy_counts.get(k, 0) + v

        # accumulate repair counts
        for k, v in repair_count.items():
            total_repair_counts[k] = total_repair_counts.get(k, 0) + v

        current_cost = pulp.value(model.objective)

        bus_count = sum(
            var.varValue for var in model.variables()
            if var.varValue is not None
        )

        buses_history.append(bus_count)
        costs_history.append(current_cost)
        
        if not improved:
            print("No improving column found.")
            break

    print(f"Total columns generated: {len(columns)}")


    # solve final master directly on columns
    final_model = solve_final_integer_master(
        trips,
        columns,
        time_limit=300,
        msg=True
    )


    # count buses
    bus_count = sum(
        1 for var in final_model.variablesDict().values()
        if var.varValue is not None and var.varValue > 0.5
    )

    # total cost
    cost = pulp.value(final_model.objective)



    # print selected columns
    for name, var in final_model.variablesDict().items():
        if var.varValue is not None and var.varValue > 0.5:
            print(name, columns[name]["trips"])

    trip_time = {
        row.trip_number: row.end_time_min - row.start_time_min
        for row in trips.itertuples(index=False)
    }
                
    stats = compute_solution_times(final_model, columns, graph, trip_time)

    print("\nTime statistics:")
    print(f"Trip time:        {stats['trip_time']:.2f} min")
    print(f"Deadhead time:    {stats['deadhead_time']:.2f} min")
    print(f"Waiting time:     {stats['waiting_time']:.2f} min")
    print(f"Total time:       {stats['total_time']:.2f} min")

    print("\nBest solution found:")
    print(f"Total trips: {len(trips)}")
    print(f"Buses used: {bus_count}")
    print(f"Total cost: {cost:.2f}")
    # print("Unique trip numbers:", trips["trip_number"].nunique())

    running_time = perf_counter() - start

    print(f"CPU Time: {running_time:.4f} seconds")

    # save_viz_data(
    #     buses=buses_history,
    #     costs=costs_history,
    #     total_cost=cost,
    #     dataset=dataset,
    #     max_iters=max_iters,
    #     total_buses=bus_count,
    #     runtime=running_time
    # )



    print("\nTotal destroy operator usage:")
    for k, v in total_destroy_counts.items():
        print(f"{k}: {v}")

    print("\nTotal repair operator usage:")
    for k, v in total_repair_counts.items():
      print(f"{k}: {v}")

if __name__ == '__main__':
    main()
