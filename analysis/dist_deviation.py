import os
import csv
import math
import argparse
import numpy as np

from ray_field.algorithm import Algorithm
from analysis import ALGORITHM_LIST, model_checkpoint_dict, class_dict

SAMPLES = 60
MODEL_NAME = 'dragon'
DIR_PATH = 'analysis/data/dist_deviation/'

def compute_values(algorithm: type[Algorithm]) -> tuple[list[int], list[list[float]]]:
    distances, R_values = algorithm.dist_deviation(model_checkpoint_dict[MODEL_NAME], 10, SAMPLES)

    return R_values, distances

def save_raw_results(R_values: list[int], distances: list[list[float]], algorithm: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    with open(f'{DIR_PATH}{MODEL_NAME}_{algorithm}_raw.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(R_values)

        for entry in distances:
            writer.writerow(entry)

def save_results(R_values: list[int], distances: list[list[float]], algorithm: str) -> None:
    with open(f'{DIR_PATH}{MODEL_NAME}_{algorithm}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        distances = np.array(distances) * 100
        means = np.mean(distances, axis=1)
        stds = np.std(distances, axis=1)
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

def load_raw_results(algorithm: str) -> tuple[list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{MODEL_NAME}_{algorithm}_raw.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        R_values = [int(value) for value in rows[0]]
        distances = [[float(value) for value in row] for row in rows[1:]]
    
    return R_values, distances

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-a", "--Algorithm", type=str)

args = parser.parse_args()

if not args.Algorithm:
    print('An algorithm must be specified')
    exit()

if args.Algorithm not in ALGORITHM_LIST:
    print(f'"{args.Algorithm}" is not a valid key, valid keys are: {", ".join(ALGORITHM_LIST)}')
    exit()

if args.Load:
    R_values, distances = load_raw_results(args.Algorithm)
    save_results(R_values, distances, args.Algorithm)
elif args.Save:
    R_values, distances = compute_values(class_dict[args.Algorithm])
    save_raw_results(R_values, distances, args.Algorithm)
    save_results(R_values, distances, args.Algorithm)
else:
    print('Neither "-s" or "-l" specified')
