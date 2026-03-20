from combine_data import parse_dhd, parse_parameters, parse_trips
from utils import (
    feasible_arcs, 
    build_deadhead_dict, 
    init_columns, 
    build_trip_graph_from_arcs_df, 
    col_gen_step,
)
from agent import save_policy, load_policy, get_env_dummy
from dataclasses import dataclass
from time import perf_counter
import pandas as pd
import pulp
import os
import datetime
import json

print("Script started")
    
def solve_final_integer_master(trips, columns):
    model = pulp.LpProblem("FinalMaster", pulp.LpMinimize)
    x = {name: pulp.LpVariable(name, cat="Binary") for name in columns}
    model += pulp.lpSum(columns[name]["cost"] * x[name] for name in columns)

    trip_ids = trips.trip_number.unique().tolist()  # FIXED: added .unique()
    for t in trip_ids:
        model += pulp.lpSum(x[name] for name, col in columns.items() if t in col["trips"]) == 1, f"cover_{t}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    model.solve(solver)
    return model, {name: pulp.value(x[name]) for name in columns}



@dataclass
class BusParams:
    fixed_cost: float = 244.13
    cost_per_km: float = 0.13
    battery_capacity_kwh: float = 160.0
    energy_per_km: float = 1.9
    max_charge_rate_kwh_min: float = 7.5
    electricity_price: float = 0.20

    @property
    def full_recharge_cost(self):
        return self.battery_capacity_kwh * self.electricity_price

    @classmethod
    def from_params_df(cls, params_df):
        row = params_df.mode().iloc[0]  # most common values
        return cls(
            fixed_cost=row['cost_per_bus'],
            cost_per_km=row['cost_per_km'],
            battery_capacity_kwh=row['battery_capacity_kwh'],
            energy_per_km=row['energy_compsumtion_kwh/km'],
            max_charge_rate_kwh_min=row['max_charge_rate_kwh/min'],
        )

def save_viz_data(buses: list[float], costs: list[float], total_cost: float, total_buses: int, dataset_name: str, runtime: float):    
    viz_data_path = f'viz_data/{dataset_name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.json'
    
    viz_data = {'steps': [], 'total_cost': total_cost, 'total_buses': total_buses, 'runtime': runtime}
    for i in range(len(buses)):
        viz_data['steps'].append({
            'step': i,
            'buses': buses[i],
            'cost': costs[i]
        })
        
    with open(viz_data_path, 'w') as f:
        json.dump(viz_data, f, indent=4)

def main():
    dataset_name = "qlink_3,7,8"
    start = perf_counter()
    UTR_trips = f'{dataset_name}/trips.txt'
    trips = parse_trips(UTR_trips)
    UTR_dhd = f'{dataset_name}/dhd.txt'
    dhd = parse_dhd(UTR_dhd)
    UTR_params = f'{dataset_name}/parameters.txt'
    params = parse_parameters(UTR_params)
    bus_params = BusParams.from_params_df(params)
    save_policy_to = f'{dataset_name}_ckeckpoint.pt'
    
    deadheads = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, deadheads, depot_stop = "nwggar", max_wait = 240)
    arcs_df = pd.DataFrame(arcs)
    graph = build_trip_graph_from_arcs_df(trips, arcs_df)
    columns = init_columns(trips)
    max_iter = 500
    
    _, state_dim, n_actions = get_env_dummy(trips, graph, arcs_df)
    pull_out = arcs_df[arcs_df['arc_type'] == 'pull_out']['to_stop'].unique()

    policy_net, optimizer, stats = load_policy(state_dim, n_actions, filepath=save_policy_to)
    
    
    var_costs = []
    var_buses = []
    for i in range(max_iter):
        improved, model, columns = col_gen_step(
            trips, graph, columns, arcs_df,bus_params=bus_params,
            policy_net=policy_net, optimizer=optimizer, stats=stats
        )
        var_costs.append(pulp.value(model.objective))
        var_buses.append(sum(1 for var in model.variablesDict().values() if var.varValue > 0.5))
        
        if i % 5 == 0:
            save_policy(policy_net, optimizer, stats, filepath=save_policy_to)

        if not improved:
            break
    
    save_policy(policy_net, optimizer, stats, filepath=save_policy_to)

    final_model, selected = solve_final_integer_master(trips, columns)
    print(f"Total number of trips to service: {len(trips['trip_number'].tolist())}")
    n_buses = [name for name, val in selected.items() if val and val > 0.5]
    total_cost = pulp.value(final_model.objective)
    print(f"Total cost: €{total_cost:.2f}")
    print(f"Buses needed: {len(n_buses)}")
    runtime = perf_counter() - start
    print(f"Total CPU time:{runtime:.4f} seconds.")
    save_viz_data(var_buses, var_costs, total_cost, len(n_buses), dataset_name, runtime)

if __name__ == '__main__':
    main()

