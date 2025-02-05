import open3d as o3d
from ray_field import prescan_cone

generated_mesh = prescan_cone.prescan_cone('bunny', 400)
o3d.visualization.draw_geometries([generated_mesh])
