import open3d as o3d
from ray_field import baseline

generated_mesh = baseline.baseline('dragon', 600)
o3d.visualization.draw_geometries([generated_mesh])
