import utils
import ray_field

import trimesh
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from trimesh import Trimesh
from numpy import ndarray

N_POINTS = 800
DATA_DIR = 'poisson_depth'
OBJECT_NAME = 'spot'

def plot_results(x: list[int], y: list[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_ylim([0, 0.3])
    ax.set_ylabel('chamfer distance')
    ax.set_xlabel('poisson depth')
    ax.set_title(OBJECT_NAME)
    fig.savefig(f'data/{DATA_DIR}/{OBJECT_NAME}_plot.svg')
    plt.show()

def ray_field_polygonization() -> tuple[list[int], list[float]]:
    original_mesh: Trimesh = ray_field.get_scaled_mesh(f'models/{OBJECT_NAME}.obj')
    rays: ndarray = ray_field.generate_rays_between_sphere_points(N_POINTS)
    intersect_locations, intersect_normals = ray_field.ray_intersection_with_mesh_batched(rays, original_mesh)

    depths: list[int] = list(range(2, 15))
    distances: list[float] = []

    for depth in depths:
        print(f'Depth:\t\t\t{depth}')

        generated_mesh = ray_field.poisson_surface_reconstruction(intersect_locations, intersect_normals, depth)
        generated_points = np.asarray(generated_mesh.vertices)

        print(f'Generated points:\t{generated_points.shape[0]}')

        original_points = trimesh.sample.sample_surface(original_mesh, generated_points.shape[0])[0]
        distance: float = utils.chamfer_distance(original_points, generated_points)

        print(f'Distance:\t\t{distance}')

        distances.append(distance)
        print("=====================")
    
    # Visualize generated mesh
    o3d.visualization.draw_geometries([generated_mesh])

    return depths, distances

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    depths, distances = ray_field.load_results(DATA_DIR, OBJECT_NAME)
    plot_results(depths, distances)
elif args.Save:
    depths, distances = ray_field_polygonization()
    ray_field.save_results(depths, distances, DATA_DIR, OBJECT_NAME)
    plot_results(depths, distances)