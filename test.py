from ray_field import prescan_cone
import open3d as o3d

mesh = prescan_cone.prescan_cone('Bunny', 600)
o3d.visualization.draw_geometries([mesh])