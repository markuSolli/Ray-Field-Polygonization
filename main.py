import utils
import trimesh
import numpy as np
import open3d as o3d

from trimesh import Trimesh
from numpy import ndarray, float64

# Read mesh from file
mesh: Trimesh = trimesh.load_mesh('suzanne.obj')

# Scale down mesh to fit unit circle
scale: ndarray[float64] = mesh.extents
transform: ndarray[float64] = trimesh.transformations.scale_matrix(2.0 / np.max(scale))
mesh.apply_transform(transform)

# Generate points along the unit sphere
sphere_points: ndarray = utils.generate_equidistant_sphere_points(100, 1.0)

# Generate rays between all points
rays: ndarray = utils.generate_rays_between_points(sphere_points)

# Perform ray intersections on the mesh
locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=rays[:, 0], ray_directions=rays[:, 1])

# Get face normals for intersection points
normals: ndarray = mesh.face_normals[index_tri]

# Create Open3D point cloud with normals
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(locations)
point_cloud.normals = o3d.utility.Vector3dVector(normals)

# Run Poisson Surface Reconstruction
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    generated_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)

generated_mesh.compute_vertex_normals()
generated_mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

# Compute Chamfer distance
distance: float = utils.chamfer_distance(mesh.vertices, generated_mesh.vertices)
print(f'Chamfer distance: {distance}')

# Visualize
o3d.visualization.draw_geometries([generated_mesh])
