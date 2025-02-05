from ray_field import baseline, utils
import trimesh
import mesh_to_sdf
import numpy as np

from ifield.data.stanford import read as stanford_read

N_SAMPLES = 30000

print('Start')
stanford_mesh = stanford_read.read_mesh('lucy')
print('Mesh read')
stanford_mesh = mesh_to_sdf.scale_to_unit_sphere(stanford_mesh)
print('Mesh scaled')

generated_mesh = baseline.baseline('lucy', 600)
print('New mesh generated')

stanford_samples = trimesh.sample.sample_surface_even(stanford_mesh, N_SAMPLES)[0]
generated_samples = np.asarray(generated_mesh.sample_points_uniformly(N_SAMPLES).points)
print('Surfaces sampled')

distance: float = utils.chamfer_distance(stanford_samples, generated_samples)
print(distance)
