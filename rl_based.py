from typing import Dict, List, Tuple, Any
from combine_data import parse_dhd, parse_parameters, parse_trips
from utils import feasible_arcs, init_columns, build_trip_graph, col_gen_step, build_deadhead_dict
import pprint
import random
import numpy as np
import pandas as pd
import pulp

def main():
    UTR_trips = 'utr/trips.txt'
    trips = parse_trips(UTR_trips)
    UTR_dhd = 'utr/dhd.txt'
    dhd = parse_dhd(UTR_dhd)
    UTR_params = 'utr/parameters.txt'
    params = parse_parameters(UTR_params)
    dhd_dict = build_deadhead_dict(dhd)
    arcs = feasible_arcs(trips, dhd_dict, max_wait=240)
    graph = build_trip_graph(trips, arcs)
    columns = init_columns(trips)
    max_iter=30
    
    for i in range(max_iter):
        improved, model, columns = col_gen_step(trips, graph, columns)
        if not improved:
            break
    
    print(f"Final Solution: {pulp.value(model.objective):2f} buses")

if __name__ == '__main__':
    main()

