import utils
import ray_field

import trimesh
import numpy as np
import open3d as o3d

from trimesh import Trimesh, Scene
from numpy import ndarray, float64

SPHERE_RADIUS = 1.0
N_POINTS_1 = 200
N_POINTS_2 = 100
N_POINTS_3 = 600
POISSON_DEPTH = 8
OBJECT_NAME = 'rocker-arm'

# Load mesh
original_mesh: Trimesh = ray_field.get_scaled_mesh(f'models/{OBJECT_NAME}.obj')
transform: ndarray[float64] = trimesh.transformations.rotation_matrix(np.pi / 4.0, [0.0, 1.0, 0.0])
original_mesh.apply_transform(transform)
scene: Scene = trimesh.Scene([original_mesh])
scene.show()

# Generate points along the unit sphere
sphere_points: ndarray = utils.generate_equidistant_sphere_points(N_POINTS_1, SPHERE_RADIUS)
point_cloud = trimesh.points.PointCloud(sphere_points, colors=[0, 0, 255])
scene = trimesh.Scene([original_mesh, point_cloud])
scene.show()

# Generate rays between all points
sphere_points = utils.generate_equidistant_sphere_points(N_POINTS_2, SPHERE_RADIUS)
rays: ndarray = ray_field.generate_rays_between_sphere_points(N_POINTS_2)
paths = []
index = 6
init_point = (sphere_points.shape[0] - 1) * index
for i in range(init_point, init_point + sphere_points.shape[0] - 1):
    paths.append(trimesh.load_path([rays[i, 0], rays[i, 0] + rays[i, 1] / 2.0]))
scene = trimesh.Scene([original_mesh, point_cloud, paths])
scene.show()

# Perform ray intersections on the mesh
intersect_locations, intersect_normals = ray_field.ray_intersection_with_mesh_batched(rays, original_mesh)
point_cloud = trimesh.points.PointCloud(intersect_locations, colors=intersect_normals / 1.2)
scene = trimesh.Scene([point_cloud])
scene.show()

# Run Poisson Surface Reconstruction
rays = ray_field.generate_rays_between_sphere_points(N_POINTS_3)
intersect_locations, intersect_normals = ray_field.ray_intersection_with_mesh_batched(rays, original_mesh)
generated_mesh = ray_field.poisson_surface_reconstruction(intersect_locations, intersect_normals, POISSON_DEPTH)
o3d.visualization.draw_geometries([generated_mesh])
