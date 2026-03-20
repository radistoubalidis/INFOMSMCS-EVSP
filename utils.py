from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import random
import pulp
import math
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
    model = pulp.LpProblem("FinalMaster", pulp.LpMinimize)
    x = {name: pulp.LpVariable(name, cat="Binary") for name in columns}

    model += pulp.lpSum(columns[name]["cost"] * x[name] for name in columns)

    trip_ids = trips.trip_number.drop_duplicates().tolist()
    for t in trip_ids:
        model += pulp.lpSum(
            x[name] for name, col in columns.items() if t in col["trips"]
        ) == 1, f"cover_{t}"

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
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
    graph = {t: [] for t in trips.trip_number.unique()}

    compat_df = arcs_df[arcs_df["arc_type"] == "tripDH"]

    for row in compat_df.itertuples(index=False):
        graph[row.from_stop].append((row.to_stop, row.travel_time, row.slack, row.distance_km))

    return graph

def build_arc_lookup_from_graph(graph):
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

    return (
        bus_params.fixed_cost
        + total_distance * bus_params.cost_per_km
        + battery_pct * bus_params.full_recharge_cost
    )
def init_columns(trips, fixed_cost):
    cols = {}
    for t in trips.trip_number:
        cols[f"col_{t}"] = {
            "trips": [t],
            "cost": fixed_cost
        }
    return cols

def solve_master(trips, columns):
    model = pulp.LpProblem("Master", pulp.LpMinimize)
    x = {name: pulp.LpVariable(name, lowBound=0) for name in columns}
    
    model += pulp.lpSum(columns[name]['cost'] * x[name] for name in columns)
    
    trip_ids = trips.trip_number.drop_duplicates().tolist()
    for t in trip_ids:
        model += pulp.lpSum(x[name] for name,col in columns.items() if t in col['trips']) == 1, f"cover_{t}"
    
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    duals = {t: model.constraints[f"cover_{t}"].pi for t in trip_ids}
    return model, duals


def col_gen_step(trips, graph, columns, pull_out_energy, pull_in_energy, bus_params):
    model, duals = solve_master(trips, columns)
    current_objective = pulp.value(model.objective)

    trip_km = {row.trip_number: row.distance_km for row in trips.itertuples(index=False)}

    candidate_blocks = []
    main_block = []
    main_rc = float("inf")


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

        if sig in existing_sigs:
            continue
        # ADD BATTERY FEASIBILITY
        if rc >= -1e-3:
            continue

        block_cost = compute_block_cost(block, graph, trip_km, bus_params, pull_out_energy, pull_in_energy)

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

class ALNSPricingEnv:
    def __init__(self, trips_df: pd.DataFrame, graph: Dict[int, List[Tuple[int, float, float, float]]],
        duals: Dict[int, float], pull_out_energy, pull_in_energy, max_iter: int = 20,
        candidate_pool_size: int = 30, reaction_factor: float = 0.2, segment_length: int = 10, seed: int | None = None, bus_params = None,
    ):
        self.trips = trips_df.set_index("trip_number")
        self.graph = graph
        self.duals = duals
        self.pull_out_energy = pull_out_energy
        self.pull_in_energy = pull_in_energy
        self.bus_params = bus_params
        
        # ALNS controls
        self.max_iter = max_iter
        self.candidate_pool_size = candidate_pool_size
        self.reaction_factor = reaction_factor
        self.segment_length = segment_length
        self.random = random.Random(seed)

        # find trip distances
        self.trip_km = {
            row.trip_number: row.distance_km
            for row in trips_df.itertuples(index=False)
        }

        self.trip_ids = trips_df.trip_number.tolist()

        # sucessor lookup: trip i -> set of feasible next trips
        self.successors = {i: {j for j, _, _, _ in nbrs} for i, nbrs in graph.items()}

        # deadhead lookup
        self.arc_lookup = build_arc_lookup_from_graph(graph)

        # destroy operators for ALNS to remove part of block
        self.destroy_ops = {
            "random_remove": self.destroy_random,
            "segment_remove": self.destroy_segment,
            # "worst_remove": self.destroy_worst,
        }
 
        # repair operators for ALNS to rebuild block after destruction
        self.repair_ops = {
            "greedy_insert": self.repair_first_improving,
            "best_insert": self.repair_best_insertion,
            # "regret2_insert": self.repair_regret2,

        }

        # ALNS adaptive weights: initially all weigths are the same, later ALNS adapts the weights
        self.destroy_weights = {k: 1.0 for k in self.destroy_ops}
        self.repair_weights = {k: 1.0 for k in self.repair_ops}

        self.destroy_scores = {k: 0.0 for k in self.destroy_ops}
        self.repair_scores = {k: 0.0 for k in self.repair_ops}

        self.destroy_attempts = {k: 0 for k in self.destroy_ops}
        self.repair_attempts = {k: 0 for k in self.repair_ops}


    def is_feasible(self, block: List[int]) -> bool:
        "Check whether all consecutive trips in the block are compatible"
        if not block:
            return False
        
        for k in range(len(block) - 1):
            if block[k + 1] not in self.successors.get(block[k], []):
                return False
        
        soc = self.bus_params.battery_capacity_kwh

        # depot -> first trip
        soc -= self.pull_out_energy.get(block[0], 0.0)
        if soc < 0:
            return False

        for i in range(len(block)):
            t = block[i]

            # trip energy
            soc -= self.trip_km[t] * self.bus_params.energy_per_km
            if soc < 0:
                return False

            # deadhead
            if i < len(block) - 1:
                j = block[i+1]

                # find distance from graph
                found_arc = False
                for nxt, _, _, dist in self.graph[t]:
                    if nxt == j:
                        soc -= dist * self.bus_params.energy_per_km
                        found_arc = True
                        break

                if not found_arc:
                    return False

                if soc < 0:
                    return False

        # last trip -> depot
        soc -= self.pull_in_energy.get(block[-1], 0.0)
        if soc < 0:
            return False
            
        return True

    def block_cost_value(self, block: List[int]) -> float:
        "Cost of one block: fixed bus cost + variable cost per km * (trip distance + deadheads)"
        return compute_block_cost(
            block,
            self.graph,
            self.trip_km,
            self.bus_params,
            self.pull_out_energy,
            self.pull_in_energy
        )

    def reduced_cost(self, block: List[int]) -> float:
        "Reduced cost = real block cost - sum of duals of covered trips"
        if not self.is_feasible(block):
            return float("inf")
        return self.block_cost_value(block) - sum(self.duals.get(t, 0.0) for t in block)


    def candidate_trips(self) -> List[int]:
        ranked = sorted(
            self.trip_ids,
            key=lambda t: self.duals.get(t, 0.0),
            reverse=True
        )

        top_k = ranked[:min(self.candidate_pool_size, len(ranked))]
        self.random.shuffle(top_k)
        return top_k

    def initial_solution(self) -> List[int]:
        candidates = self.candidate_trips()
        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda t: self.duals.get(t, 0.0),
            reverse=True
        )

        pool_size = min(50, len(ranked))
        seed_pool = ranked[:pool_size]

        n_seeds = min(10, len(seed_pool))
        seed_list = self.random.sample(seed_pool, n_seeds)

        best_block = [seed_list[0]]
        best_rc = self.reduced_cost(best_block)

        for seed in seed_list:
            block = [seed]
            improved = True

            while improved:
                improved = False
                last = block[-1]

                nbrs = list(self.successors.get(last, []))
                self.random.shuffle(nbrs)

                for nxt in nbrs:
                    if nxt in block:
                        continue

                    cand = block + [nxt]
                    if self.reduced_cost(cand) < self.reduced_cost(block):
                        block = cand
                        improved = True
                        break

            rc = self.reduced_cost(block)
            if rc < best_rc:
                best_block = block
                best_rc = rc

        return best_block

    def destroy_random(self, block: List[int], q: int = None) -> List[int]:
        "Randomly remove one or two trips from the current block"
        if len(block) <= 1:
            return block.copy()

        if q is None:
            q = 1 if len(block) < 4 else 2

        q = min(q, len(block) - 1)
        idxs = sorted(self.random.sample(range(len(block)), q), reverse=True)

        partial = block.copy()
        for idx in idxs:
            partial.pop(idx)

        return partial

    def destroy_segment(self, block: List[int], q: int = None) -> List[int]:
        "Remove one contiguous segment from the current block"
        if len(block) <= 1:
            return block.copy()

        if q is None:
            q = 1 if len(block) < 4 else 2

        q = min(q, len(block) - 1)
        start = self.random.randint(0, len(block) - q)

        return block[:start] + block[start + q:]
    
    def destroy_worst(self, block: List[int], q: int = None) -> List[int]:
        " Remove trips with lowest dual values (worst) from the current block"
        if len(block) <= 1:
            return block.copy()

        if q is None:
            q = 1 if len(block) < 4 else 2

        q = min(q, len(block) - 1)

        scored = [(self.duals.get(t, 0.0), t) for t in block]
        scored.sort()  # lowest dual first

        to_remove = {t for _, t in scored[:q]}

        return [t for t in block if t not in to_remove]

    def repair_first_improving(self, partial_block: List[int]) -> List[int]:
        "Repair a partial block by trying to insert trips in a random order, stops at immediately if reduced cost improves"
        candidates = self.candidate_trips()
        current = partial_block.copy()
        remaining = [t for t in candidates if t not in current]

        if not current and remaining:
            best_single = min(remaining, key=lambda t: self.reduced_cost([t]))
            current = [best_single]
            remaining.remove(best_single)

        improved = True
        while improved:
            improved = False
            base_rc = self.reduced_cost(current)

            # randomize candidate order
            self.random.shuffle(remaining)

            for trip in remaining[:]:
                positions = list(range(len(current) + 1))
                self.random.shuffle(positions)

                for pos in positions:
                    cand = current[:pos] + [trip] + current[pos:]
                    if not self.is_feasible(cand):
                        continue

                    cand_rc = self.reduced_cost(cand)
                    if cand_rc < base_rc:
                        current = cand
                        remaining.remove(trip)
                        improved = True
                        break

                if improved:
                    break

        return current
    
    def repair_best_insertion(self, partial_block: List[int]) -> List[int]:
        "Repair a partial block by repeatedly choosing best feasible insertion among all possibilities"
        candidates = self.candidate_trips()
        current = partial_block.copy()
        remaining = [t for t in candidates if t not in current]

        if not current and remaining:
            best_single = min(remaining, key=lambda t: self.reduced_cost([t]))
            current = [best_single]
            remaining.remove(best_single)

        improved = True
        while improved:
            improved = False
            base_rc = self.reduced_cost(current)

            best_trip = None
            best_pos = None
            best_rc = base_rc

            for trip in remaining:
                for pos in range(len(current) + 1):
                    cand = current[:pos] + [trip] + current[pos:]
                    if not self.is_feasible(cand):
                        continue

                    cand_rc = self.reduced_cost(cand)
                    if cand_rc < best_rc:
                        best_rc = cand_rc
                        best_trip = trip
                        best_pos = pos

            if best_trip is not None:
                current = current[:best_pos] + [best_trip] + current[best_pos:]
                remaining.remove(best_trip)
                improved = True

        return current
    
    def repair_regret2(self, partial_block: List[int]) -> List[int]:
        "Repair a partial block using regret-2 insertion, for each trip compare its best and second best insertion cost."
        "Trips with high regret = second best - best, are inserted first because postponing them would be bad"

        candidates = self.candidate_trips()
        current = partial_block.copy()
        remaining = [t for t in candidates if t not in current]

        # If empty, start with best singleton
        if not current and remaining:
            best_single = min(remaining, key=lambda t: self.reduced_cost([t]))
            current = [best_single]
            remaining.remove(best_single)

        improved = True
        while improved and remaining:
            improved = False
            base_rc = self.reduced_cost(current)

            best_trip = None
            best_pos = None
            best_regret = -float("inf")
            best_new_rc = float("inf")

            for trip in remaining:
                insertion_values = []

                for pos in range(len(current) + 1):
                    cand = current[:pos] + [trip] + current[pos:]
                    if not self.is_feasible(cand):
                        continue

                    cand_rc = self.reduced_cost(cand)
                    insertion_values.append((cand_rc, pos))

                if not insertion_values:
                    continue

                insertion_values.sort(key=lambda x: x[0])

                best_val, best_pos_trip = insertion_values[0]
                second_val = insertion_values[1][0] if len(insertion_values) > 1 else best_val

                regret = second_val - best_val

                # choose highest regret; break ties by best resulting reduced cost
                if regret > best_regret or (regret == best_regret and best_val < best_new_rc):
                    best_regret = regret
                    best_trip = trip
                    best_pos = best_pos_trip
                    best_new_rc = best_val

            if best_trip is not None and best_new_rc < base_rc:
                current = current[:best_pos] + [best_trip] + current[best_pos:]
                remaining.remove(best_trip)
                improved = True

        return current
    
    
    def select_operator(self, weights: Dict[str, float]) -> str:
        "Select a destroy or repair operator using adaptive weights"
        names = list(weights.keys())
        probs = np.array([weights[n] for n in names], dtype=float)
        probs = probs / probs.sum()

        return self.random.choices(names, weights=probs, k=1)[0]

    def accept(self, s: List[int], s_new: List[int], temperature: float) -> bool:
        "Acceptance rule for ALNS, using simulated annealing"

        rc_s = self.reduced_cost(s)
        rc_new = self.reduced_cost(s_new)

        if rc_new < rc_s:
            return True
        # if new block is worse, accept with some probability to escape local minima
        if temperature <= 1e-12:
            return False

        delta = rc_new - rc_s

        return self.random.random() < math.exp(-delta / temperature)

    def update_scores(self, destroy_name: str, repair_name: str, outcome: str):
        "Reward operators depending on how good the accepted move was"
        if outcome == "global_best":
            reward = 5.0
        elif outcome == "improved":
            reward = 3.0
        elif outcome == "accepted":
            reward = 1.0
        else:
            reward = 0.0

        self.destroy_scores[destroy_name] += reward
        self.repair_scores[repair_name] += reward
        self.destroy_attempts[destroy_name] += 1
        self.repair_attempts[repair_name] += 1

    def update_weights(self):
        "Update operator weights using average score over recent attempts"
        lam = self.reaction_factor

        for name in self.destroy_ops:
            if self.destroy_attempts[name] > 0:
                avg_score = self.destroy_scores[name] / self.destroy_attempts[name]
                self.destroy_weights[name] = max(
                    0.1, (1 - lam) * self.destroy_weights[name] + lam * avg_score
                )

        for name in self.repair_ops:
            if self.repair_attempts[name] > 0:
                avg_score = self.repair_scores[name] / self.repair_attempts[name]
                self.repair_weights[name] = max(
                    0.1, (1 - lam) * self.repair_weights[name] + lam * avg_score
                )

    def reset_scores(self):
        "Reset the ALNS weights after a segment of iterations"
        self.destroy_scores = {k: 0.0 for k in self.destroy_ops}
        self.repair_scores = {k: 0.0 for k in self.repair_ops}

    def reset_attempts(self):
        "Reset the ALNS weights after a segment of iterations"
        self.destroy_attempts = {k: 0 for k in self.destroy_ops}
        self.repair_attempts = {k: 0 for k in self.repair_ops}

    def solve(self) -> Tuple[List[int], float]:
        "Run ALNS until max_iter or until a negative reduced costs block is found"
        s = self.initial_solution()
        if not s:
            return [], float("inf")

        s_best = s.copy()
        best_rc = self.reduced_cost(s_best)
        temperature = 3.0

        for it in range(1, self.max_iter + 1):
            # choose operators
            d_name = self.select_operator(self.destroy_weights)
            r_name = self.select_operator(self.repair_weights)

            destroy = self.destroy_ops[d_name]
            repair = self.repair_ops[r_name]
            
            # apply destroy and repair
            s_new = repair(destroy(s))

            # compute reduced costs
            rc_old = self.reduced_cost(s)
            rc_new = self.reduced_cost(s_new)

            if self.accept(s, s_new, temperature):
                s = s_new

                if rc_new < best_rc:
                    s_best = s_new.copy()
                    best_rc = rc_new
                    self.update_scores(d_name, r_name, "global_best")
                elif rc_new < rc_old:
                    self.update_scores(d_name, r_name, "improved")
                else:
                    self.update_scores(d_name, r_name, "accepted")
            else:
                self.update_scores(d_name, r_name, "rejected")

            # early stop: pricing only needs one negative reduced-cost column
            # if best_rc < -1e-3:
            #     return s_best, best_rc
            
            if it >= 5 and best_rc < -1e-3:
                return s_best, best_rc
            
            

            # update the weight every segment
            if it % self.segment_length == 0:
                self.update_weights()
                self.reset_scores()
                self.reset_attempts()

            # decrease the temperature to not less worse solutions over time
            temperature *= 0.99

        return s_best, best_rc

def generate_contiguous_subcolumns(block, duals, graph, trip_km, bus_params, pull_out_energy, pull_in_energy, min_len=2):
    "Generate all contiguous subcolumns from a block"
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