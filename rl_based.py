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
import pandas as pd
import pulp
print("Script started")
    
def solve_final_integer_master(trips, columns):
    model = pulp.LpProblem("FinalMaster", pulp.LpMinimize)
    x = {name: pulp.LpVariable(name, cat="Binary") for name in columns}
    model += pulp.lpSum(columns[name]["cost"] * x[name] for name in columns)

    trip_ids = trips.trip_number.unique().tolist()  # FIXED: added .unique()
    for t in trip_ids:
        model += pulp.lpSum(x[name] for name, col in columns.items() if t in col["trips"]) == 1, f"cover_{t}"

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return model



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


def main():
    DEBUG = False
    if DEBUG:
        UTR_trips = 'utr/trips.txt'
        trips = parse_trips(UTR_trips)
        UTR_dhd = 'utr/dhd.txt'
        dhd = parse_dhd(UTR_dhd)
        UTR_params = 'utr/parameters.txt'
        params = parse_parameters(UTR_params)
        bus_params = BusParams.from_params_df(params)
        save_policy_to = 'debug_policy_ckeckpoint.pt'
    else:
        trips = pd.read_csv('combined_datasets/trips.csv')
        trips['trip_number'] = trips['line_number'].astype(str) + '_' + trips['trip_number'].astype(str)
        dhd = pd.read_csv('combined_datasets/dhd.csv')
        params = pd.read_csv('combined_datasets/parameters.csv')
        bus_params = BusParams.from_params_df(params)
        save_policy_to = 'policy_ckeckpoint.pt'
    
    deadheads = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, deadheads, depot_stop = "utrgar", max_wait = 1000)
    arcs_df = pd.DataFrame(arcs)
    graph = build_trip_graph_from_arcs_df(trips, arcs_df)
    columns = init_columns(trips)
    max_iter = 100
    
    _, state_dim, n_actions = get_env_dummy(trips, graph, arcs_df)
    pull_out = arcs_df[arcs_df['arc_type'] == 'pull_out']['to_stop'].unique()

    policy_net, optimizer, stats = load_policy(state_dim, n_actions, filepath=save_policy_to)
    
    for i in range(max_iter):
        improved, model, columns = col_gen_step(
            trips, graph, columns, arcs_df,bus_params=bus_params,
            policy_net=policy_net, optimizer=optimizer, stats=stats
        )
        
        if i % 5 == 0:
            save_policy(policy_net, optimizer, stats, filepath=save_policy_to)
        
        if not improved:
            break
    
    save_policy(policy_net, optimizer, stats, filepath=save_policy_to)

    final_model = solve_final_integer_master(trips, columns)
    print(f"Total number of trips to service: {len(trips['trip_number'].tolist())}")
    n_buses = sum(1 for var in final_model.variablesDict().values() if var.varValue > 0.5)
    total_cost = pulp.value(final_model.objective)
    print(f"Total cost: €{total_cost:.2f}")
    print(f"Buses needed: {n_buses}")

if __name__ == '__main__':
    main()

