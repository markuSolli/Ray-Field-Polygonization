import os
import csv
import argparse
import math
import numpy as np

from ray_field import CheckpointName
from ray_field.baseline_device import BaselineDevice
from analysis import N_VALUES, OBJECT_NAMES, model_checkpoint_dict

SAMPLES = 60
DIR_PATH = 'analysis/data/time_deviation/'

def compute_values(model_name: CheckpointName) -> tuple[list[int], list[list[float]]]:
    # Warm up
    BaselineDevice.surface_reconstruction(model_name, 100)

    times, R_values = BaselineDevice.time_deviation(model_name, N_VALUES, SAMPLES)

    return R_values, times

def save_raw_results(R_values: list[int], times: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    with open(f'{DIR_PATH}{model_name}_raw.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(R_values)

        for entry in times:
            writer.writerow(entry)

def save_results(R_values: list[int], times: list[list[float]], model_name: str) -> None:
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        times = np.array(times)
        means = np.mean(times, axis=1)
        stds = np.std(times, axis=1)
        L = means * 0.05
        n = ((2 * 1.96 * stds) / L) ** 2

        writer.writerow(['|R|', 'Mean', 'Std', 'L', 'N'])

        for r, mean, std, l, nn in zip(R_values, means, stds, L, n):
            writer.writerow([
                r,
                round(mean, 3),
                round(std, 3),
                round(l, 3),
                math.ceil(nn)
            ])

def load_raw_results(model_name: str) -> tuple[list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}_raw.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        R_values = [int(value) for value in rows[0]]
        times = [[float(value) for value in row] for row in rows[1:]]
    
    return R_values, times

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-f", "--Filename", type=str)

args = parser.parse_args()

if not args.Filename:
    print('A filename must be specified')
    exit()

if args.Filename not in OBJECT_NAMES:
    print(f'"{args.Filename}" is not a valid key, valid keys are: {", ".join(OBJECT_NAMES)}')
    exit()

if args.Load:
    R_values, times = load_raw_results(args.Filename)
    save_results(R_values, times, args.Filename)
elif args.Save:
    R_values, times = compute_values(model_checkpoint_dict[args.Filename])
    save_raw_results(R_values, times, args.Filename)
    save_results(R_values, times, args.Filename)
else:
    print('Neither "-s" or "-l" specified')