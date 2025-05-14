import os
import csv
import argparse
import numpy as np

from ray_field import CheckpointName
from ray_field.baseline_device import BaselineDevice
from ray_field.prescan_cone import PrescanCone
from ray_field.candidate_sphere import CandidateSphere
from ray_field.angle_filter import AngleFilter
from analysis import OBJECT_NAMES, model_name_dict, model_checkpoint_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/chamfer_dist/'
ALG_NAMES = ['Baseline', 'Prescan Cone', 'Candidate Sphere', 'Angle Filter']

def compute_values(model_name: CheckpointName) -> tuple[list[str], list[list[int]], list[list[float]]]:
    distances = []
    R_values = []

    for i, algorithm in enumerate([BaselineDevice, PrescanCone, CandidateSphere, AngleFilter]):
        print(ALG_NAMES[i])
        d, R = algorithm.chamfer(model_name, 10)
        distances.append(d)
        R_values.append(R)

    return ALG_NAMES, R_values, distances

def save_results(alg_names: list[str], R_values: list[list[int]], distances: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(alg_names)
        writer.writerow([len(R_values)])
        
        for entry in R_values:
            writer.writerow(entry)

        for entry in distances:
            writer.writerow(entry)

def load_results(model_name: str) -> tuple[list[str], list[list[int]], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        alg_names = rows[0]
        length = int(rows[1][0])
        R_values = [[int(value) for value in row] for row in rows[2:2+length]]
        distances = [[float(value) for value in row] for row in rows[2+length:]]
    
    return alg_names, R_values, distances

def plot_results(alg_names: list[str], R_values: list[list[int]], distances: list[list[float]], model_name: str) -> None:
    fig, ax = plt.subplots()
    distances = np.array(distances) * 100

    for i in range(len(alg_names)):
        ax.plot(R_values[i], distances[i], label=alg_names[i])
    
    ax.set_ylabel('CD$\\cdot10^2$')
    ax.set_xlabel('|R|')
    #ax.set_ylim([0, 0.035])
    ax.set_title(f'{model_name_dict[model_name]}')
    ax.legend(loc=(1.04, 0), title='Object')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}{model_name}.png', bbox_inches="tight")

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
    alg_names, R_values, distances = load_results(args.Filename)
    plot_results(alg_names, R_values, distances, args.Filename)
elif args.Save:
    alg_names, R_values, distances = compute_values(model_checkpoint_dict[args.Filename])
    save_results(alg_names, R_values, distances, args.Filename)
    plot_results(alg_names, R_values, distances, args.Filename)
else:
    print('Neither "-s" or "-l" specified')