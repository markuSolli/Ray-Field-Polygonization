import open3d as o3d

from ray_field.candidate_sphere import CandidateSphere

mesh = CandidateSphere.surface_reconstruction('bunny', 100)
o3d.visualization.draw_geometries([mesh])
