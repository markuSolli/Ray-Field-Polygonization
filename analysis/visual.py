import argparse
import open3d as o3d
import numpy as np

from ray_field.baseline_device import BaselineDevice
from ray_field.prescan_cone import PrescanCone
from ray_field.candidate_sphere import CandidateSphere
from ray_field.angle_filter import AngleFilter
from old_analysis import ALGORITHM_LIST, OBJECT_NAMES

DIR_PATH = 'analysis/data/visual/'

camera_position = np.array([0.0, -4.0, 0.0])
look_at = np.array([0.0, 0.0, 0.0])
up_direction = np.array([0.0, 0.0, 1.0])

forward = (look_at - camera_position)
forward /= np.linalg.norm(forward)

right = np.cross(forward, up_direction)
right /= np.linalg.norm(right)

up = np.cross(right, forward)

camera_transform = np.eye(4)
camera_transform[:3, 0] = right
camera_transform[:3, 1] = up
camera_transform[:3, 2] = -forward
camera_transform[:3, 3] = camera_position

renderer = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"

def visualize_baseline(model_name: str):
    for N in [50, 250, 500]:
        mesh = BaselineDevice.surface_reconstruction(model_name, N)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
        o3d.visualization.draw_geometries([mesh],
                                            zoom=1.0,
                                            front=camera_position,
                                            lookat=look_at,
                                            up=up)

def visualize_prescan_cone(model_name: str):
    for N in [50, 950, 1900]:
        mesh = PrescanCone.surface_reconstruction(model_name, N)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
        o3d.visualization.draw_geometries([mesh],
                                                zoom=1.0,
                                                front=camera_position,
                                                lookat=look_at,
                                                up=up)

def visalize_candidate_sphere(model_name: str):
    for N in [50, 950, 1900]:
        mesh = CandidateSphere.surface_reconstruction(model_name, N)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
        o3d.visualization.draw_geometries([mesh],
                                                zoom=1.0,
                                                front=camera_position,
                                                lookat=look_at,
                                                up=up)

def visualize_angle_filter(model_name: str):
    for N in [50, 250, 500]:
        mesh = AngleFilter.surface_reconstruction(model_name, N)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
        o3d.visualization.draw_geometries([mesh],
                                                zoom=1.0,
                                                front=camera_position,
                                                lookat=look_at,
                                                up=up)

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--Algorithm", type=str)
parser.add_argument("-f", "--Filename", type=str)

args = parser.parse_args()

if not args.Algorithm:
    print('An algorithm must be specified')
    exit()
elif not args.Filename:
    print('A filename must be specified')
    exit()

if args.Algorithm not in ALGORITHM_LIST:
    print(f'"{args.Algorithm}" is not a valid key, valid keys are: {", ".join(ALGORITHM_LIST)}')
    exit()
elif args.Filename not in OBJECT_NAMES:
    print(f'"{args.Filename}" is not a valid key, valid keys are: {", ".join(OBJECT_NAMES)}')
    exit()
else:
    if args.Algorithm == 'baseline':
        visualize_baseline(args.Filename)
    elif args.Algorithm == 'prescan_cone':
        visualize_prescan_cone(args.Filename)
    elif args.Algorithm == 'candidate_sphere':
        visalize_candidate_sphere(args.Filename)
    elif args.Algorithm == 'angle_filter':
        visualize_angle_filter(args.Filename)
