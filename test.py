from ray_field import algorithms
import open3d as o3d

mesh = algorithms.baseline_prescan('Bunny', 600)
o3d.visualization.draw_geometries([mesh])