import utils

import csv
import trimesh
import numpy as np
import open3d as o3d

from trimesh import Trimesh
from numpy import ndarray, float64
from open3d.geometry import TriangleMesh
from open3d.utility import VerbosityLevel

SPHERE_RADIUS = 1.0
BATCH_SIZE = 10000

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

def ray_intersection_with_mesh_batched(rays: ndarray, mesh: Trimesh) -> tuple[ndarray, ndarray]:
    # Initialize lists to hold results
    all_locations = []
    all_normals = []

    num_rays = rays.shape[0]

    # Process rays in batches
    for start in range(0, num_rays, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_rays)

        # Perform ray intersections on the mesh
        locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=rays[start:end, 0], ray_directions=rays[start:end, 1])

        # Get face normals for intersection points
        normals: ndarray = mesh.face_normals[index_tri]

        # Append results to the lists
        all_locations.append(locations)
        all_normals.append(normals)
    
    # Concatenate all results into final arrays
    final_locations = np.concatenate(all_locations, axis=0)
    final_normals = np.concatenate(all_normals, axis=0)

    return final_locations, final_normals


def poisson_surface_reconstruction(points: ndarray, normals: ndarray, depth: int, verbosity: VerbosityLevel = VerbosityLevel.Error) -> TriangleMesh:
    # Create Open3D point cloud with normals
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    # Run Poisson Surface Reconstruction
    with o3d.utility.VerbosityContextManager(verbosity) as cm:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)[0]

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

    return mesh

def save_results(x: list[int], y: list[float], dir: str, obj_name: str) -> None:
    with open(f'data/{dir}/{obj_name}_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        for x_value, y_value in zip(x, y):
            writer.writerow([x_value, y_value])

def load_results(dir: str, obj_name: str) -> tuple[list[int], list[float]]:
    with open(f'data/{dir}/{obj_name}_data.csv', mode='r') as file:
        reader = csv.reader(file)
        n_points = []
        distances = []

        for row in reader:
            n_points.append(int(row[0]))
            distances.append(float(row[1]))
    
    return n_points, distances