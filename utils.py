from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import random
import pulp
import math

# def feasible_arcs(trips: pd.DataFrame, deadheads: pd.DataFrame, max_wait = 10):
#     arcs = []

#     for i in trips.itertuples():
#         for j in trips.itertuples():
#             if i.trip_number == j.trip_number:
#                 continue
            
#             if i.to_stop == j.from_stop:
#                 travel_time = 0
#             else:

#                 dh = deadheads[(deadheads.from_stop == i.to_stop) & (deadheads.to_stop == j.from_stop)]

#                 if dh.empty:
#                     continue

#                 travel_time = dh.iloc[0]["time_ver_0"]

#             slack = j.start_time_min - (i.end_time_min + travel_time) # type:ignore

#             if 0 <= slack <= max_wait: # type:ignore
#                 arcs.append((i.trip_number, j.trip_number))

#     return arcs


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
    graph = {t: [] for t in trips.trip_number.unique() if t not in [1190,1192,1194]}

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
def compute_block_cost(block, graph, trip_km, fixed_cost=244.13, cost_per_km=0.13):
    if len(block) == 0:
        return 0.0
    if len(block) == 1:
        return fixed_cost

    arc_lookup = build_arc_lookup_from_graph(graph)

    total_km = 0.0

    # add service trip km
    for t in block:
        total_km += trip_km[t]

        # add deadhead km between consecutive trips
    for k in range(len(block) - 1):
        i, j = block[k], block[k + 1]
        if (i, j) not in arc_lookup:
            return float("inf")
        total_km += arc_lookup[(i, j)]["distance_km"]

    return fixed_cost + cost_per_km * total_km

    # for k in range(len(block) - 1):
    #     i, j = block[k], block[k + 1]
    #     if (i, j) not in arc_lookup:
    #         return float("inf")

    #     travel_time = arc_lookup[(i, j)]["travel_time"]
    #     slack = arc_lookup[(i, j)]["slack"]
    #     total_time_cost += (travel_time + slack) * time_cost_per_min


    # return fixed_cost + total_time_cost

def init_columns(trips):
    cols = {}
    for t in trips.trip_number:
        cols[f"col_{t}"] = {
            "trips": [t],
            "cost": 1000.0
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


def col_gen_step(trips, graph, columns, sub_problem="rl"):
    model, duals = solve_master(trips, columns)
    current_objective = pulp.value(model.objective)

    trip_km = {row.trip_number: row.distance_km for row in trips.itertuples(index=False)}

    if sub_problem == "rl":
        env = EVSPPricingEnv(trips_df=trips, graph=graph, duals=duals)
        block, reduced_cost = greedy_pi_policy(env)
        new_blocks = [(block, reduced_cost)] if block and reduced_cost < -1e-3 else []

    elif sub_problem == "metaheuristics":
        new_blocks = pricing_multi_columns(
            trips=trips,
            graph=graph,
            duals=duals
        )

    else:
        raise ValueError(f"Unknown sub_problem: {sub_problem}")

    if not new_blocks:
        print("No improvement found!")
        return False, model, columns

    existing_sigs = {tuple(col["trips"]) for col in columns.values()}
    added = 0

    for block, reduced_cost in new_blocks:
        sig = tuple(block)

        if sig in existing_sigs:
            continue
        if reduced_cost >= -1e-3:
            continue

        block_cost = compute_block_cost(block, graph, trip_km)
        col_name = f"col_{len(columns)}"
        columns[col_name] = {
            "trips": block,
            "cost": block_cost
        }

        existing_sigs.add(sig)
        added += 1

    if added == 0:
        print("No new distinct improving columns added!")
        return False, model, columns

    return True, model, columns

class EVSPPricingEnv:
    def __init__(self, trips_df: pd.DataFrame, graph: Dict[int, List[int]], 
                 duals: Dict[int, float], block_cost: float = 1000.0, cost_per_km: float = 0.1, max_len: int = 30):
        self.trips = trips_df.set_index('trip_number')
        self.graph = graph
        self.duals = duals
        self.block_cost = block_cost
        self.cost_per_km = cost_per_km
        self.max_len = max_len
        
        # All possible trips
        self.all_trips = sorted(list(graph.keys()))
        self.trip_to_idx = {t: i for i, t in enumerate(self.all_trips)}
        self.n_trips = len(self.all_trips)
        
        self.reset()
    
    def reset(self) -> Tuple[int, np.ndarray]:
        """Start new block (before any trip)"""
        self.current_trip = None
        self.block = []         
        self.visited_mask = np.zeros(self.n_trips, dtype=bool)
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> Tuple[int, np.ndarray]:
        """State = (current_trip_idx, visited_mask)"""
        curr_idx = -1 if self.current_trip is None else self.trip_to_idx[self.current_trip]
        return curr_idx, self.visited_mask.copy()
    
    def get_actions(self) -> List[Any]:
        """Possible actions: next trips + STOP"""
        if self.current_trip is None:
            # Can start at any unvisited trip
            available = [t for t in self.all_trips if not self.visited_mask[self.trip_to_idx[t]]]
        else:
            # Successors of current trip
            available = [t for t in self.graph[self.current_trip] 
                        if not self.visited_mask[self.trip_to_idx[t]]]
        
        available.append("STOP")
        return available
    
    def step(self, action: Any) -> Tuple[Tuple[int, np.ndarray], float, bool, Dict]:
        """Take action, return next_state, reward, done, info"""
        self.steps += 1
        reward = 0.0
        done = False
        info = {}
        
        if action == "STOP" or self.steps >= self.max_len:
            # Terminal: compute reduced cost
            sum_pi = sum(self.duals[t] for t in self.block)
            reduced_cost = self.block_cost - sum_pi
            # Reward = -(block_cost - sum(π_t)) = sum(π_t) - block_cost
            reward = -reduced_cost  
            done = True
            info = {
                "block": self.block.copy(),
                "sum_pi": sum_pi,
                "reduced_cost": reduced_cost,
                "block_length": len(self.block)
            }
        else:
            # Add trip to block
            self.current_trip = action
            self.block.append(action)
            trip_idx = self.trip_to_idx[action]
            self.visited_mask[trip_idx] = True
        
        return self._get_state(), reward, done, info


def random_policy(env) -> Tuple[List[int], float]:
    """Random policy for testing (replace later with trained RL)"""
    state = env.reset()
    done = False
    
    while not done:
        actions = env.get_actions()
        action = random.choice(actions)
        state, reward, done, info = env.step(action)
    
    return info["block"], info["reduced_cost"]

def greedy_pi_policy(env) -> Tuple[List[int], float]:
    """Greedy: always pick highest π_t successor"""
    state = env.reset()
    done = False
    
    while not done:
        actions = [a for a in env.get_actions() if a != "STOP"]
        if not actions:
            # No more trips, stop
            _, reward, done, info = env.step("STOP")
            break
            
        # Pick highest π_t
        action = max(actions, key=lambda t: env.duals[t])
        state, reward, done, info = env.step(action)
    
        state, _, done, info = env.step(action)
    
    return info['block'], info['reduced_cost']

class ALNSPricingEnv:
    def __init__(
        self,
        trips_df: pd.DataFrame,
        graph: Dict[int, List[Tuple[int, float, float, float]]],
        duals: Dict[int, float],
        block_cost: float = 1000.0,
        time_cost_per_min: float = 0.1,
        max_iter: int = 40,
        candidate_pool_size: int = 30,
        reaction_factor: float = 0.2,
        segment_length: int = 10,
        seed: int = 42,
    ):
        self.trips = trips_df.set_index("trip_number")
        self.graph = graph
        self.duals = duals
        self.block_cost = block_cost
        self.time_cost_per_min = time_cost_per_min
        self.max_iter = max_iter
        self.candidate_pool_size = candidate_pool_size
        self.reaction_factor = reaction_factor
        self.segment_length = segment_length
        self.random = random.Random(seed)
        self.trip_km = {
        row.trip_number: row.distance_km
        for row in trips_df.itertuples(index=False)
        }
        self.cost_per_km = 0.13

        self.trip_ids = trips_df.trip_number.tolist()
        self.successors = {i: {j for j, _, _, _ in nbrs} for i, nbrs in graph.items()}
        self.arc_lookup = build_arc_lookup_from_graph(graph)

        # lightweight neighborhoods
        self.destroy_ops = {
            "random_remove": self.destroy_random,
            "segment_remove": self.destroy_segment,
        }

        self.repair_ops = {
            "greedy_insert": self.repair_first_improving,
        }

        self.destroy_weights = {k: 1.0 for k in self.destroy_ops}
        self.repair_weights = {k: 1.0 for k in self.repair_ops}

        self.destroy_scores = {k: 0.0 for k in self.destroy_ops}
        self.repair_scores = {k: 0.0 for k in self.repair_ops}

        self.destroy_attempts = {k: 0 for k in self.destroy_ops}
        self.repair_attempts = {k: 0 for k in self.repair_ops}

    def is_feasible(self, block: List[int]) -> bool:
        if not block:
            return False
        for k in range(len(block) - 1):
            if block[k + 1] not in self.successors.get(block[k], set()):
                return False
        return True

    def block_cost_value(self, block: List[int]) -> float:
        return compute_block_cost(
            block,
            self.graph,
            trip_km=self.trip_km,
            fixed_cost=self.block_cost,
            cost_per_km=self.cost_per_km
        )

    def reduced_cost(self, block: List[int]) -> float:
        if not self.is_feasible(block):
            return float("inf")
        return self.block_cost_value(block) - sum(self.duals.get(t, 0.0) for t in block)

    def candidate_trips(self) -> List[int]:
        return self.trip_ids.copy()
    
    def initial_solution(self) -> List[int]:
        candidates = self.candidate_trips()
        if not candidates:
            return []

        # try several seeds, keep the best block found
        seed_list = sorted(candidates, key=lambda t: self.duals.get(t, 0.0), reverse=True)[:10]
        best_block = [seed_list[0]]
        best_rc = self.reduced_cost(best_block)

        for seed in seed_list:
            block = [seed]

            improved = True
            while improved:
                improved = False
                last = block[-1]

                nbrs = list(self.successors.get(last, set()))
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
        if len(block) <= 1:
            return block.copy()

        if q is None:
            q = 1 if len(block) < 4 else 2

        q = min(q, len(block) - 1)
        start = self.random.randint(0, len(block) - q)
        return block[:start] + block[start + q:]

    def repair_first_improving(self, partial_block: List[int]) -> List[int]:
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

            # randomize candidate order a bit
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

    def select_operator(self, weights: Dict[str, float]) -> str:
        names = list(weights.keys())
        probs = np.array([weights[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        return self.random.choices(names, weights=probs, k=1)[0]

    def accept(self, s: List[int], s_new: List[int], temperature: float) -> bool:
        rc_s = self.reduced_cost(s)
        rc_new = self.reduced_cost(s_new)

        if rc_new < rc_s:
            return True
        if temperature <= 1e-12:
            return False

        delta = rc_new - rc_s
        return self.random.random() < math.exp(-delta / temperature)

    def update_scores(self, destroy_name: str, repair_name: str, outcome: str):
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
        self.destroy_scores = {k: 0.0 for k in self.destroy_ops}
        self.repair_scores = {k: 0.0 for k in self.repair_ops}

    def reset_attempts(self):
        self.destroy_attempts = {k: 0 for k in self.destroy_ops}
        self.repair_attempts = {k: 0 for k in self.repair_ops}

    def solve(self) -> Tuple[List[int], float]:
        # Initialization
        s = self.initial_solution()
        if not s:
            return [], float("inf")

        s_best = s.copy()
        best_rc = self.reduced_cost(s_best)
        temperature = 3.0

        for it in range(1, self.max_iter + 1):
            d_name = self.select_operator(self.destroy_weights)
            r_name = self.select_operator(self.repair_weights)

            destroy = self.destroy_ops[d_name]
            repair = self.repair_ops[r_name]

            s_new = repair(destroy(s))
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
            if best_rc < -1e-3:
                return s_best, best_rc

            if it % self.segment_length == 0:
                self.update_weights()
                self.reset_scores()
                self.reset_attempts()

            temperature *= 0.99

        return s_best, best_rc


def alns_pricing(trips, graph, duals):
    pricer = ALNSPricingEnv(
        trips_df=trips,
        graph=graph,
        duals=duals,
        block_cost=1000.0,
        time_cost_per_min=0.1,
        max_iter=40,
        candidate_pool_size=30,
        reaction_factor=0.2,
        segment_length=10,
    )
    return pricer.solve()

def column_signature(block):
    return tuple(block)


def existing_column_signatures(columns):
    return {tuple(col["trips"]) for col in columns.values()}

def generate_short_columns(trips, graph, duals, max_pairs_per_seed=3, use_triples=False, max_triples_total=200):
    """
    Generate promising 2-trip columns, and optionally a limited number of 3-trip columns.
    Returns list of (block, reduced_cost).
    """
    results = []
    trip_ids = trips.trip_number.tolist()

    def reduced_cost(block):
        cost = compute_block_cost(block, graph)
        return cost - sum(duals.get(t, 0.0) for t in block)

    # 2-trip columns
    for i in trip_ids:
        succs = graph.get(i, [])
        scored_pairs = []

        for j, travel_time, slack, distance_km in succs:
            block = [i, j]
            rc = reduced_cost(block)
            if rc < -1e-6:
                scored_pairs.append((block, rc))

        scored_pairs.sort(key=lambda x: x[1])
        results.extend(scored_pairs[:max_pairs_per_seed])

    # Optional, heavily capped 3-trip columns
    if use_triples:
        triple_count = 0
        for i in trip_ids:
            succs_i = graph.get(i, [])

            for j, _, _, _ in succs_i[:max_pairs_per_seed]:
                succs_j = graph.get(j, [])

                for k, _, _, _ in succs_j[:max_pairs_per_seed]:
                    if k == i:
                        continue

                    block = [i, j, k]
                    rc = reduced_cost(block)
                    if rc < -1e-6:
                        results.append((block, rc))
                        triple_count += 1

                    if triple_count >= max_triples_total:
                        break
                if triple_count >= max_triples_total:
                    break
            if triple_count >= max_triples_total:
                break

    # deduplicate
    best_by_block = {}
    for block, rc in results:
        sig = tuple(block)
        if sig not in best_by_block or rc < best_by_block[sig]:
            best_by_block[sig] = rc

    final_results = [(list(sig), rc) for sig, rc in best_by_block.items()]
    final_results.sort(key=lambda x: x[1])
    return final_results

def alns_pricing_multi(trips, graph, duals, n_runs=8, max_cols=10):
    """
    Run lightweight ALNS multiple times with different seeds and collect distinct negative columns.
    Returns list of (block, reduced_cost).
    """
    found = {}

    for run in range(n_runs):
        pricer = ALNSPricingEnv(
            trips_df=trips,
            graph=graph,
            duals=duals,
            block_cost=1000.0,
            time_cost_per_min=0.1,
            max_iter=20,
            candidate_pool_size=len(trips),
            reaction_factor=0.2,
            segment_length=10,
            seed=42 + run,
        )

        block, rc = pricer.solve()

        if block and rc < -1e-6:
            sig = tuple(block)
            if sig not in found or rc < found[sig]:
                found[sig] = rc

    results = [(list(sig), rc) for sig, rc in found.items()]
    results.sort(key=lambda x: x[1])  # most negative first
    return results[:max_cols]

def pricing_multi_columns(trips, graph, duals):
    candidates = []

    # use your real trip distance column name here
    trip_km = {row.trip_number: row.distance_km for row in trips.itertuples(index=False)}

    print("  pricing: start ALNS multi")
    alns_cols = alns_pricing_multi(trips, graph, duals, n_runs=4, max_cols=5)
    print(f"  pricing: ALNS multi returned {len(alns_cols)} columns")

    for block, rc in alns_cols:
        if not block:
            continue

        candidates.append((block, rc))

        sub_cols = generate_contiguous_subcolumns(block, duals, graph, trip_km, min_len=2)
        candidates.extend(sub_cols)

    best_by_block = {}
    for block, rc in candidates:
        sig = tuple(block)
        if sig not in best_by_block or rc < best_by_block[sig]:
            best_by_block[sig] = rc

    results = [(list(sig), rc) for sig, rc in best_by_block.items()]
    results.sort(key=lambda x: x[1])

    print(f"  pricing: total distinct improving columns {len(results)}")
    return results

def generate_contiguous_subcolumns(block, duals, graph, trip_km, min_len=6):
    """
    Generate all contiguous feasible subcolumns from a block.
    Since the original block is feasible, every contiguous slice is feasible.
    Returns list of (subblock, reduced_cost).
    """
    results = []
    n = len(block)

    for start in range(n):
        for end in range(start + min_len, n + 1):
            sub = block[start:end]
            cost = compute_block_cost(sub, graph, trip_km)
            rc = cost - sum(duals.get(t, 0.0) for t in sub)

            if rc < -1e-6:
                results.append((sub, rc))

    return results

