import argparse
import trimesh
import torch
from ray_field import utils, baseline, prescan_cone
from analysis import ALGORITHM_LIST, OBJECT_NAMES
import open3d as o3d

from trimesh import Scene
from numpy import ndarray

N_POINTS = 200

def visualize_baseline(model_name: str):
    # Load mesh
    stanford_mesh = utils.load_and_scale_stanford_mesh(model_name)
    scene: Scene = trimesh.Scene([stanford_mesh])
    scene.show()   

    # Generate points along the unit sphere
    sphere_points: ndarray = utils.generate_equidistant_sphere_points(N_POINTS)
    point_cloud = trimesh.points.PointCloud(sphere_points, colors=[0, 0, 255])
    scene = trimesh.Scene([stanford_mesh, point_cloud])
    scene.show()

    # Generate rays between all points
    origins, dirs = utils.generate_rays_between_points(sphere_points)
    j = 14
    paths = []
    for i in range(dirs.shape[2]):
        paths.append(trimesh.load_path([origins[j], origins[j] + dirs[j,0,i] / 2.0]))
    scene = trimesh.Scene([stanford_mesh, point_cloud, paths])
    scene.show()

    # Perform ray intersections on the mesh
    model, device = utils.init_model(model_name)
    origins = origins.to(device)
    dirs = dirs.to(device)
    intersections, intersection_normals = baseline.baseline_scan(model, origins, dirs)
    point_cloud = trimesh.points.PointCloud(intersections, colors=intersection_normals / 1.2)
    scene = trimesh.Scene([point_cloud])
    scene.show()

    # Run Poisson Surface Reconstruction
    generated_mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, 8)
    o3d.visualization.draw_geometries([generated_mesh])

def visualize_prescan_cone(model_name: str):
    PRESCAN_N = 100

    # Load mesh
    stanford_mesh = utils.load_and_scale_stanford_mesh(model_name)
    scene: Scene = trimesh.Scene([stanford_mesh])
    scene.show()   

    # Broad scan
    model, device = utils.init_model(model_name)
    origins, dirs = utils.generate_sphere_rays(device, PRESCAN_N)
    intersections = prescan_cone.prescan_cone_broad_scan(model, origins, dirs).cpu().detach().numpy()
    intersect_cloud = trimesh.points.PointCloud(intersections, colors=[0, 0, 255])
    scene = trimesh.Scene([stanford_mesh, intersect_cloud])
    scene.show()

    # Generate cone rays
    sphere_points: ndarray = utils.generate_equidistant_sphere_points(N_POINTS)
    point_cloud = trimesh.points.PointCloud(sphere_points, colors=[0, 0, 255])
    intersections = torch.from_numpy(intersections).to(device)
    origins, dirs = prescan_cone.generate_cone_rays(intersections, N_POINTS, device)
    origins = origins.cpu().detach().numpy()
    dirs = dirs.cpu().detach().numpy()
    j = 0
    paths = []
    for i in range(dirs.shape[2]):
        paths.append(trimesh.load_path([origins[j], origins[j] + dirs[j,0,i] / 2.0]))
    scene = trimesh.Scene([stanford_mesh, point_cloud, paths])
    scene.show()

    # Targeted scan
    origins = torch.from_numpy(origins).to(device)
    dirs = torch.from_numpy(dirs).to(device)
    intersections, intersection_normals = prescan_cone.prescan_cone_targeted_scan(model, origins, dirs)
    point_cloud = trimesh.points.PointCloud(intersections, colors=intersection_normals / 1.2)
    scene = trimesh.Scene([point_cloud])
    scene.show()

    # Run Poisson Surface Reconstruction
    generated_mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, 8)
    o3d.visualization.draw_geometries([generated_mesh])

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
