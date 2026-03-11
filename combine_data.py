import pandas as pd
from io import StringIO
import argparse
import os

def parse_trips(path, verbose=0):
    with open(path, 'r', encoding='cp1252', errors='replace') as f:
        t_lines = [line for line in f if line.startswith('T;')]
    
    df_trips = pd.read_csv(StringIO("".join(t_lines)), sep=';', header=None, engine='python') # type:ignore
    
    # select only columns with useful information
    df_trips = df_trips.iloc[:, [0,1,2,5,6,7,8,-3]]
    
    df_trips.columns = [
        "record_type",
        "line_number",
        "trip_number",
        # "trip_direction",
        # "variant_code",
        "from_stop",
        "start_time_min",
        "end_time_min",
        "to_stop",
        "distance_km"
    ]
    
    df_trips['distance_km'] = pd.to_numeric(df_trips['distance_km'], errors='coerce')
    if verbose == 1:
        print(f'Parsed {path}')
        print(df_trips.head())
    return df_trips

def parse_dhd(path, verbose=0):
    # read text file and only keep deadhead records
    with open(path, "r", encoding="cp1252", errors="replace") as f:
        t_lines = [line for line in f if line.startswith("D;")]

    # parse the deadhead records as pandas df
    df_deadheads = pd.read_csv(StringIO("".join(t_lines)), sep=";", header=None, engine="python")

    # select only columns with useful information
    df_deadheads = df_deadheads.iloc[:, [0,1,2,3,4,5,6]]

    df_deadheads.columns = [
        "record_type",
        "from_stop-to_stop",
        "time_ver_0",
        "time_ver_1",
        "time_ver_2",
        "time_ver_3",
        "distance_km",
    ]
    df_deadheads["distance_km"] = pd.to_numeric(df_deadheads["distance_km"], errors="coerce")
    df_deadheads[["from_stop", "to_stop"]] = df_deadheads["from_stop-to_stop"].str.split("-", n=1, expand=True)
    
    if verbose == 1:
        print(f"Parsed {path}")
        print(df_deadheads.head())
    return df_deadheads

def parse_parameters(path, verbose=0):
    with open(path, "r", encoding="cp1252", errors="replace") as f:
        t_lines = [line for line in f if line.startswith("U;")]

    # read text file and only keep parameters records
    df_params = pd.read_csv(StringIO("".join(t_lines)), sep=";", header=None, engine="python")

    # parse the parameters records as pandas df
    df_params = df_params.iloc[:, [0,2,4,9,10,12]]

    # select only columns with useful information
    df_params.columns = [
        "record_type",
        "cost_per_bus",
        # "cost_per_minute",
        "cost_per_km",
        "energy_compsumtion_kwh/km",
        "max_charge_rate_kwh/min",
        "battery_capacity_kwh"
    ]
    
    if verbose == 1:
        print(f"Parsed {path}")
        print(df_params.head())
    return df_params


def main():
    data_dirs = ['gn12', 'gn345', 'qlink_3,7,8', 'qlink_3', 'qlink_7,8', 'qlink_7', 'qlink_8', 'utr']
    parser = argparse.ArgumentParser(description='Process a directory')
    parser.add_argument('directory', help='Path to the input directory')
    args = parser.parse_args()

    dirs_to_read_from = [os.path.join(args.directory, data_dir) for data_dir in os.listdir(args.directory) if data_dir in data_dirs]
    
    trips_all = []
    dhd_all = []
    parameters_all = []
    
    for dataset in dirs_to_read_from:
        trips_all.append(parse_trips(f"{dataset}/trips.txt"))
        dhd_all.append(parse_dhd(f"{dataset}/dhd.txt"))
        parameters_all.append(parse_parameters(f"{dataset}/parameters.txt"))
    
    trips = pd.concat(trips_all, ignore_index=True)
    dhd = pd.concat(dhd_all, ignore_index=True)
    parameters = pd.concat(parameters_all, ignore_index=True)
    
    store_to = "combined_datasets"
    trips.to_csv(f"{store_to}/trips.csv", index=False)
    dhd.to_csv(f"{store_to}/dhd.csv", index=False)
    parameters.to_csv(f"{store_to}/parameters.csv", index=False)
    
    
if __name__ == '__main__':
    main()