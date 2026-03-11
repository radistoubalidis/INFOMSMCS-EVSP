from combine_data import parse_dhd, parse_parameters, parse_trips
from utils import feasible_arcs, EVSPPricingEnv, random_policy, greedy_pi_policy
import pprint
import pandas as pd
import pulp


def build_trip_graph(trips: pd.DataFrame, arcs):
    """
    arcs: list[(i_trip, j_trip)] from feasible_arcs
    returns adjacency list: {trip: [successor_trips]}
    """
    graph = {t: [] for t in trips.trip_number.unique()}
    for i, j in arcs:
        graph[i].append(j)
    return graph



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
    block, reduced_cost = greedy_pi_policy(env)
    
    print(f"Current obj: {current_objective:.2f}, Found block: {block}, reduced_cost: {reduced_cost:.3f}")
    
    if reduced_cost >= -1e-6:
        print("No improvement found!")
        return False, model, columns
    
    col_name = f"col_{len(columns)}"
    columns[col_name] = {
        "trips": block,
        "cost": 1.0
    }
    print(f"Added {col_name}: {block}")
    return True, model, columns


def main():
    # UTR_trips = 'evsp-instances/trips.txt'
    # trips = parse_trips(UTR_trips)
    # UTR_dhd = 'evsp-instances/dhd.txt'
    # dhd = parse_dhd(UTR_dhd)
    # UTR_params = 'evsp-instances/parameters.txt'
    # params = parse_parameters(UTR_params)
    trips = pd.read_csv('combined_datasets/trips.csv')
    dhd = pd.read_csv('combined_datasets/dhd.csv')
    parameters = pd.read_csv('combined_datasets/parameters.csv')
    arcs = feasible_arcs(trips, dhd, max_wait=120)
    graph = build_trip_graph(trips, arcs)
    columns = init_columns(trips)
    max_wait=60
    max_iter=20
    
    for i in range(max_iter):
        improved, model, columns = col_gen_step(trips, graph, columns)
        if not improved:
            break
    
    print(f"Final Solution: {pulp.value(model.objective):2f} buses")

if __name__ == '__main__':
    main()

