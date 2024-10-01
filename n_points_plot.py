import utils
import ray_field

import trimesh
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from trimesh import Trimesh
from numpy import ndarray

POISSON_DEPTH = 8
DATA_DIR = 'n_points'
OBJECT_NAME = 'suzanne'

def plot_results(x: list[int], y: list[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_ylim([0, 0.04])
    ax.set_ylabel('chamfer distance')
    ax.set_xlabel('n')
    ax.set_title(OBJECT_NAME)
    fig.savefig(f'data/{DATA_DIR}/{OBJECT_NAME}_plot.svg')
    plt.show()

def ray_field_polygonization() -> tuple[list[int], list[float]]:
    original_mesh: Trimesh = ray_field.get_scaled_mesh(f'models/{OBJECT_NAME}.obj')

    n_points: list[int] = list(range(100, 1001, 100))
    distances: list[float] = []

    for n in n_points:
        print(f'Points:\t\t\t{n}')

        rays: ndarray = ray_field.generate_rays_between_sphere_points(n)

        print(f'Rays:\t\t\t{rays.shape[0]}')

        intersect_locations, intersect_normals = ray_field.ray_intersection_with_mesh_batched(rays, original_mesh)
        generated_mesh = ray_field.poisson_surface_reconstruction(intersect_locations, intersect_normals, POISSON_DEPTH)

        generated_points = np.asarray(generated_mesh.vertices)

        print(f'Generated points:\t{generated_points.shape[0]}')

        original_points = trimesh.sample.sample_surface(original_mesh, generated_points.shape[0])[0]
        distance: float = utils.chamfer_distance(original_points, generated_points)

        print(f'Distance:\t\t{distance}')

        distances.append(distance)
        print("=====================")
    
    # Visualize generated mesh
    o3d.visualization.draw_geometries([generated_mesh])

    return n_points, distances

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    n_points, distances = ray_field.load_results(DATA_DIR, OBJECT_NAME)
    plot_results(n_points, distances)
elif args.Save:
    n_points, distances = ray_field_polygonization()
    ray_field.save_results(n_points, distances, DATA_DIR, OBJECT_NAME)
    plot_results(n_points, distances)