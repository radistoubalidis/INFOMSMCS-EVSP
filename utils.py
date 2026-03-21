import pandas as pd
import pulp
from dataclasses import dataclass

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
    
def solve_final_integer_master(trips, columns, time_limit=60, msg=False):
    "Solve final master with integer result"

    model = pulp.LpProblem("FinalMaster", pulp.LpMinimize) # objective: minimize
    x = {name: pulp.LpVariable(name, cat="Binary") for name in columns} # each column is selected or not

    model += pulp.lpSum(columns[name]["cost"] * x[name] for name in columns)

    trip_ids = trips.trip_number.drop_duplicates().tolist()
    for t in trip_ids:
        model += pulp.lpSum(x[name] for name, col in columns.items() if t in col["trips"]) == 1, f"cover_{t}" # every trip covered exactly once

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit) # solve using time limit
    model.solve(solver)
    return model

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
def feasible_arcs(trips: pd.DataFrame, deadhead_dict, depot_stop="utrgar", max_wait=240):
    arcs = []
    trips_list = list(trips.itertuples())

    # feasible arcs depot -> trips
    for j in trips_list:
        if depot_stop == j.from_stop:
            travel_time = 0
            distance_km = 0.0
        else:
            key = (depot_stop, j.from_stop)
            if key not in deadhead_dict:
                continue

            dh = deadhead_dict[key]
            travel_time = dh["t1"]
            distance_km = dh["distance_km"]

        arcs.append({
            "arc_type": "pull_out",
            "from_stop": "DEPOT",
            "to_stop": j.trip_number,
            "travel_time": travel_time,
            "distance_km": distance_km,
            "slack": None
        })

    # feasible arcs trips -> trips
    for i in trips_list:
        for j in trips_list:
            if i.trip_number == j.trip_number:
                continue

            if i.to_stop == j.from_stop:
                travel_time = 0
                distance_km = 0.0
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
                    "distance_km": distance_km,
                    "slack": slack
                })

    # feasible arcs trips -> depot
    for i in trips_list:
        if i.to_stop == depot_stop:
            travel_time = 0
            distance_km = 0.0
        else:
            key = (i.to_stop, depot_stop)
            if key not in deadhead_dict:
                continue

            dh = deadhead_dict[key]
            travel_time = get_deadhead_time(i.end_time_min, dh)
            distance_km = dh["distance_km"]

        arcs.append({
            "arc_type": "pull_in",
            "from_stop": i.trip_number,
            "to_stop": "DEPOT",
            "travel_time": travel_time,
            "distance_km": distance_km,
            "slack": None
        })

    return arcs

def build_trip_graph_from_arcs_df(trips: pd.DataFrame, arcs_df: pd.DataFrame):
    "Build the trip graph using only inter-trip deadheads"
    graph = {t: [] for t in trips.trip_number.unique()}

    compat_df = arcs_df[arcs_df["arc_type"] == "tripDH"]

    for row in compat_df.itertuples(index=False):
        graph[row.from_stop].append((row.to_stop, row.travel_time, row.slack, row.distance_km))

    return graph

def build_arc_lookup_from_graph(graph):
    "Build arc lookup"
    arc_lookup = {}
    for i, nbrs in graph.items():
        for j, travel_time, slack, distance_km in nbrs:
            arc_lookup[(i, j)] = {
                "travel_time": travel_time,
                "slack": slack,
                "distance_km": distance_km
            }
    return arc_lookup


# Column Generation Part
def compute_block_cost(block, graph, trip_km, bus_params, pull_out_energy=None, pull_in_energy=None):
    "Compute cost of one block"

    if len(block) == 0:
        return 0.0

    trip_distance = sum(trip_km[t] for t in block)

    deadhead_distance = 0.0
    for k in range(len(block) - 1):
        arc = next((a for a in graph.get(block[k], []) if a[0] == block[k + 1]), None)
        if arc is None:
            return float("inf")
        deadhead_distance += arc[3]

    # depot pull-out / pull-in energy in kWh
    pull_out_kwh = 0.0 if pull_out_energy is None else pull_out_energy.get(block[0], 0.0)
    pull_in_kwh = 0.0 if pull_in_energy is None else pull_in_energy.get(block[-1], 0.0)

    # convert depot energy back to distance-equivalent for km cost
    pull_out_distance = pull_out_kwh / bus_params.energy_per_km
    pull_in_distance = pull_in_kwh / bus_params.energy_per_km

    # total distance includes depot travel
    total_distance = trip_distance + deadhead_distance + pull_out_distance + pull_in_distance

    total_energy = total_distance * bus_params.energy_per_km

    battery_pct = min(total_energy / bus_params.battery_capacity_kwh, 1.0)

    # return the fixed costs per bus + (total distance * cost per km) + (percentage battery used * recharge cost)
    return (
        bus_params.fixed_cost
        + total_distance * bus_params.cost_per_km
        + battery_pct * bus_params.full_recharge_cost
    )

def init_columns(trips, fixed_cost):
    "Initial columns"
    cols = {}
    for t in trips.trip_number:
        cols[f"col_{t}"] = {
            "trips": [t],
            "cost": fixed_cost
        }
    return cols

def solve_master(trips, columns):
    "Solve Restricted Master Problem as an LP"
    model = pulp.LpProblem("Master", pulp.LpMinimize) # objective: minimize costs
    x = {name: pulp.LpVariable(name, lowBound=0) for name in columns}
    
    model += pulp.lpSum(columns[name]['cost'] * x[name] for name in columns) # one variable for each column
    
    trip_ids = trips.trip_number.drop_duplicates().tolist()
    for t in trip_ids:
        model += pulp.lpSum(x[name] for name,col in columns.items() if t in col['trips']) == 1, f"cover_{t}" # cover every trip exactly once
    
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    duals = {t: model.constraints[f"cover_{t}"].pi for t in trip_ids} # extract duals for pricing problem
    return model, duals


def col_gen_step(trips, graph, columns, pull_out_energy, pull_in_energy, bus_params):
    "Main function for one column generation iteration"
    model, duals = solve_master(trips, columns) # solve current master LP and get dual prices
    current_objective = pulp.value(model.objective)

    trip_km = {row.trip_number: row.distance_km for row in trips.itertuples(index=False)}

    candidate_blocks = []
    main_block = []
    main_rc = float("inf")


    # run ALNS to find a new block
    main_block, main_rc = alns_pricing(
        trips=trips,
        graph=graph, 
        duals=duals, 
        pull_out_energy=pull_out_energy, 
        pull_in_energy=pull_in_energy,  
        bus_params=bus_params
        )

    if main_block:
        # add main ALNS block
        candidate_blocks.append((main_block, main_rc))

        # add subcolumns from that one block
        sub_cols = generate_contiguous_subcolumns(
            main_block,
            duals,
            graph,
            trip_km,
            bus_params,
            pull_out_energy,
            pull_in_energy,
            min_len=4
        )
        candidate_blocks.extend(sub_cols)

    if not candidate_blocks:
        print("No improvement found!")
        return False, model, columns

    existing_sigs = {tuple(col["trips"]) for col in columns.values()}
    added = 0

    print(f"Current obj: {current_objective:.2f}")
    print(f"Main priced block: {main_block}, reduced_cost={main_rc:.3f}")

    for block, rc in candidate_blocks:
        sig = tuple(block)

        # check for duplicate and non-negatice reduced cost blocks
        if sig in existing_sigs:
            continue
        if rc >= -1e-3:
            continue

        # compute block cost
        block_cost = compute_block_cost(block, graph, trip_km, bus_params, pull_out_energy, pull_in_energy)

        # add new column
        col_name = f"col_{len(columns)}"
        columns[col_name] = {
            "trips": block,
            "cost": block_cost
        }

        existing_sigs.add(sig)
        added += 1

        print(f"Added {col_name}: rc={rc:.3f}, cost={block_cost:.2f}, trips={block}")

    if added == 0:
        print("No new distinct improving columns added!")
        return False, model, columns

    return True, model, columns

def generate_contiguous_subcolumns(block, duals, graph, trip_km, bus_params, pull_out_energy, pull_in_energy, min_len=2):
    "Generate all contiguous subcolumns from a block with length at least min_len"
    results = []
    n = len(block)

    for start in range(n):
        for end in range(start + min_len, n + 1):
            sub = block[start:end]
            cost = compute_block_cost(sub, graph, trip_km, bus_params, pull_out_energy, pull_in_energy)
            rc = cost - sum(duals.get(t, 0.0) for t in sub)

            if rc < -1e-6:
                results.append((sub, rc))

    return results

def alns_pricing(trips, graph, duals, pull_out_energy, pull_in_energy, bus_params):
    "Run one ALNS pricing solve and return one improving block."
    from alns_pricing_env import ALNSPricingEnv
    pricer = ALNSPricingEnv(
        trips_df=trips,
        graph=graph,
        duals=duals,
        pull_out_energy=pull_out_energy,
        pull_in_energy=pull_in_energy, 
        max_iter=20,
        candidate_pool_size=50,
        reaction_factor=0.2,
        segment_length=10,
        seed=None,
        bus_params=bus_params
    )
    return pricer.solve()