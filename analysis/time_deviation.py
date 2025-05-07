import os
import csv
import argparse
import numpy as np

from ray_field.baseline_device import BaselineDevice
from analysis import N_VALUES

SAMPLES = 60
MODEL_NAME = 'bunny'
DIR_PATH = 'analysis/data/time_deviation/'

def compute_values() -> tuple[list[int], list[list[float]]]:
    # Warm up
    BaselineDevice.surface_reconstruction(MODEL_NAME, 100)

    times, R_values = BaselineDevice.time_deviation(MODEL_NAME, N_VALUES, SAMPLES)

    return R_values, times

def save_results(R_values: list[int], times: list[list[float]]) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{MODEL_NAME}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        times = np.array(times)
        means = np.mean(times, axis=1)
        stds = np.std(times, axis=1)
        L = means * 0.05
        n = ((2 * 1.96 * stds) / L) ** 2

        writer.writerow(['|R|', 'Mean', 'Std', 'L', 'N'])

        for r, mean, std, l, nn in zip(R_values, means, stds, L, n):
            writer.writerow([r, mean, std, l, nn])

    with open(f'{DIR_PATH}{MODEL_NAME}_raw.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        for entry in times:
            writer.writerow(entry)

parser = argparse.ArgumentParser()

args = parser.parse_args()

R_values, times = compute_values()
save_results(R_values, times)