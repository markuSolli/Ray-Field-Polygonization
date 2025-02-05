from ray_field import baseline, utils
import trimesh
import mesh_to_sdf
import numpy as np

from ifield.data.stanford import read as stanford_read

N_SAMPLES = 30000

stanford_mesh = stanford_read.read_mesh('bunny')
stanford_mesh = mesh_to_sdf.scale_to_unit_sphere(stanford_mesh)

generated_mesh = baseline.baseline('bunny', 600)

stanford_samples = trimesh.sample.sample_surface_even(stanford_mesh, N_SAMPLES)[0]
generated_samples = np.asarray(generated_mesh.sample_points_uniformly(N_SAMPLES).points)

distance: float = utils.chamfer_distance(stanford_samples, generated_samples)
print(distance)
