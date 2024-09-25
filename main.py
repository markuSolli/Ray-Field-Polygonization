import utils

import csv
import trimesh
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from trimesh import Trimesh
from numpy import ndarray, float64
from open3d.geometry import TriangleMesh
from open3d.utility import VerbosityLevel

SPHERE_RADIUS = 1.0
POISSON_DEPTH = 8
RESULT_FILE = 'chamfer_data.csv'

def get_scaled_mesh(filepath: str) -> Trimesh:
    # Read mesh from file
    mesh: Trimesh = trimesh.load_mesh(filepath)

    # Scale down mesh to fit unit circle
    scale: ndarray[float64] = mesh.extents
    transform: ndarray[float64] = trimesh.transformations.scale_matrix(2.0 / np.max(scale))
    mesh.apply_transform(transform)

    return mesh

def generate_rays_between_sphere_points(n: int) -> ndarray:
    # Generate points along the unit sphere
    sphere_points: ndarray = utils.generate_equidistant_sphere_points(n, SPHERE_RADIUS)

    # Generate rays between all points
    return utils.generate_rays_between_points(sphere_points)

def ray_intersection_with_mesh(rays: ndarray, mesh: Trimesh) -> tuple[ndarray, ndarray]:
    # Perform ray intersections on the mesh
    locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=rays[:, 0], ray_directions=rays[:, 1])

    # Get face normals for intersection points
    normals: ndarray = mesh.face_normals[index_tri]

    return locations, normals

def poisson_surface_reconstruction(points: ndarray, normals: ndarray, verbosity: VerbosityLevel = VerbosityLevel.Debug) -> TriangleMesh:
    # Create Open3D point cloud with normals
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    # Run Poisson Surface Reconstruction
    with o3d.utility.VerbosityContextManager(verbosity) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=POISSON_DEPTH)

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

    return mesh

def save_results(x: list[int], y: list[float]) -> None:
    with open(RESULT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)

        for x_value, y_value in zip(x, y):
            writer.writerow([x_value, y_value])

def load_results() -> tuple[list[int], list[float]]:
    with open(RESULT_FILE, mode='r') as file:
        reader = csv.reader(file)
        n_points = []
        distances = []

        for row in reader:
            n_points.append(int(row[0]))
            distances.append(float(row[1]))
    
    return n_points, distances

def plot_results(x: list[int], y: list[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.savefig('chamfer_plot.svg')
    plt.show()

def ray_field_polygonization() -> tuple[list[int], list[float]]:
    original_mesh: Trimesh = get_scaled_mesh('suzanne.obj')

    n_points: list[int] = list(range(100, 1001, 50))
    distances: list[float] = []

    for n in n_points:
        rays: ndarray = generate_rays_between_sphere_points(n)
        intersect_locations, intersect_normals = ray_intersection_with_mesh(rays, original_mesh)
        generated_mesh = poisson_surface_reconstruction(intersect_locations, intersect_normals)
        distance: float = utils.chamfer_distance(original_mesh.vertices, generated_mesh.vertices)

        distances.append(distance)
    
    return n_points, distances

# Visualize generated mesh
#o3d.visualization.draw_geometries([generated_mesh])

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    n_points, distances = load_results()
    plot_results(n_points, distances)
elif args.Save:
    n_points, distances = ray_field_polygonization()
    save_results(n_points, distances)
    plot_results(n_points, distances)
