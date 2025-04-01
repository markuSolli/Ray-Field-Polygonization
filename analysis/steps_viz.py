import argparse
import trimesh
import torch
import open3d as o3d
import numpy as np

from ray_field import utils
from ray_field.baseline import Baseline
from ray_field.prescan_cone import PrescanCone
from ray_field.candidate_sphere import CandidateSphere
from ray_field.convex_hull import ConvexHull
from analysis import ALGORITHM_LIST, OBJECT_NAMES
from trimesh import Scene
from numpy import ndarray

N_POINTS = 200

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

def index_to_color(index: int):
    r = (index % 4) * 64
    g = (index % 6) * 43
    b = (index % 8) * 32

    return [r, g, b]

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
    model, device = utils.init_model(model_name)
    origins, dirs = utils.generate_rays_between_points(sphere_points, device)
    origins = origins.cpu().detach().numpy()
    dirs = dirs.cpu().detach().numpy()
    j = 14
    paths = []
    for i in range(dirs.shape[2]):
        paths.append(trimesh.load_path([origins[j], origins[j] + dirs[j,0,i] / 2.0]))
    scene = trimesh.Scene([stanford_mesh, point_cloud, paths])
    scene.show()

    # Perform ray intersections on the mesh
    origins = torch.from_numpy(origins).to(device)
    dirs = torch.from_numpy(dirs).to(device)
    intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)
    point_cloud = trimesh.points.PointCloud(intersections, colors=intersection_normals / 1.2)
    scene = trimesh.Scene([point_cloud])
    scene.show()

    # Run Poisson Surface Reconstruction
    generated_mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, 8)
    o3d.visualization.draw_geometries([generated_mesh])

def visualize_prescan_cone(model_name: str):
    # Load mesh
    stanford_mesh = utils.load_and_scale_stanford_mesh(model_name)
    scene: Scene = trimesh.Scene([stanford_mesh])
    scene.camera_transform = camera_transform
    scene.show()

    with torch.no_grad():
        # Broad scan
        model, device = utils.init_model(model_name)
        origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
        intersections, intersection_normals = PrescanCone._broad_scan(model, origins, dirs)
        intersections = intersections.cpu().detach().numpy()
        intersection_normals = intersection_normals.cpu().detach().numpy()
        intersect_cloud = trimesh.points.PointCloud(intersections, colors=intersection_normals / 1.2)
        scene = trimesh.Scene([intersect_cloud])
        scene.camera_transform = camera_transform
        scene.show()

        # Generate cone rays
        sphere_points: ndarray = utils.generate_equidistant_sphere_points(N_POINTS)
        point_cloud = trimesh.points.PointCloud(sphere_points, colors=[0, 0, 255])
        intersections = torch.from_numpy(intersections).to(device)
        origins, dirs = PrescanCone._generate_cone_rays(intersections, N_POINTS, device)
        origins = origins.cpu().detach().numpy()
        dirs = dirs.cpu().detach().numpy()
        j = 0
        paths = []
        for i in range(dirs.shape[2]):
            paths.append(trimesh.load_path([origins[j], origins[j] + dirs[j,0,i] / 2.0]))
        scene = trimesh.Scene([stanford_mesh, point_cloud, paths])
        scene.camera_transform = camera_transform
        scene.show()

        # Targeted scan
        origins = torch.from_numpy(origins).to(device)
        dirs = torch.from_numpy(dirs).to(device)
        intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
        intersections = intersections.cpu().detach().numpy()
        intersection_normals = intersection_normals.cpu().detach().numpy()
        point_cloud = trimesh.points.PointCloud(intersections, colors=intersection_normals / 1.2)
        scene = trimesh.Scene([point_cloud])
        scene.camera_transform = camera_transform
        scene.show()

        # Run Poisson Surface Reconstruction
        generated_mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, 8)
        o3d.visualization.draw_geometries([generated_mesh],
                                            zoom=1.0,
                                            front=camera_position,
                                            lookat=look_at,
                                            up=up)

def visalize_candidate_sphere(model_name: str):
    # Load mesh
    stanford_mesh = utils.load_and_scale_stanford_mesh(model_name)
    scene: Scene = trimesh.Scene([stanford_mesh])
    scene.camera_transform = camera_transform
    scene.show()   

    # Broad scan
    model, device = utils.init_model(model_name)
    origins, dirs = utils.generate_sphere_rays(CandidateSphere.prescan_n, device)
    intersections, intersection_normals, all_sphere_centers = CandidateSphere._broad_scan(model, origins, dirs)
    intersections = intersections.cpu().detach().numpy()
    colors = np.array([index_to_color(i) for i in range(16)], dtype=np.uint8)
    colors = np.repeat(colors[np.newaxis, :, :], intersections.shape[0], axis=0)
    intersections_flat = intersections.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    surface_cloud = trimesh.points.PointCloud(intersections_flat, colors=colors)
    scene = trimesh.Scene([surface_cloud])
    scene.camera_transform = camera_transform
    scene.show()

    # Candidate spheres
    intersections = torch.from_numpy(intersections).to(device)
    radii, centers = CandidateSphere._generate_candidate_spheres(intersections, device)
    radii = radii.cpu().detach().numpy()
    centers = centers.cpu().detach().numpy()
    
    spheres = []
    for i in range(16):
        color = index_to_color(i)
        color.append(64)

        sphere = trimesh.primitives.Sphere(radii[i], centers[i])
        sphere.visual.face_colors[:] = np.array(color)
        spheres.append(sphere)

    scene = trimesh.Scene([surface_cloud, spheres])
    scene.camera_transform = camera_transform
    scene.show()

    # Generate rays
    M = (N_POINTS - 1) // 16
    sphere_points: ndarray = utils.generate_equidistant_sphere_points(N_POINTS)
    point_cloud = trimesh.points.PointCloud(sphere_points, colors=[0, 0, 255])
    radii = torch.from_numpy(radii).to(device)
    centers = torch.from_numpy(centers).to(device)
    origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N_POINTS, device)
    origins = origins.cpu().detach().numpy()
    dirs = dirs.cpu().detach().numpy()
    k = 0
    paths = []
    for i in range(16):
        color = index_to_color(i)
        for j in range(i * M, (i + 1) * M):
            path = trimesh.load_path([origins[k], origins[k] + dirs[k,0,j] / 2.0])
            path.colors = [color] * len(path.entities)
            paths.append(path)

    scene = trimesh.Scene([stanford_mesh, point_cloud, paths])
    scene.camera_transform = camera_transform
    scene.show()

    # Targeted scan
    origins = torch.from_numpy(origins).to(device)
    dirs = torch.from_numpy(dirs).to(device)
    intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
    point_cloud = trimesh.points.PointCloud(intersections, colors=intersection_normals / 1.2)
    scene = trimesh.Scene([point_cloud])
    scene.camera_transform = camera_transform
    scene.show()

    # Poisson Surface Reconstruction
    generated_mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
    o3d.visualization.draw_geometries([generated_mesh],
                                    zoom=1.0,
                                    front=camera_position,
                                    lookat=look_at,
                                    up=up)

def visualize_convex_hull(model_name: str):
    # Broad scan
    model, device = utils.init_model(model_name)
    origins, dirs = utils.generate_sphere_rays_tensor(N_POINTS, device)
    intersections, atom_indices = ConvexHull._baseline_scan(model, origins, dirs)

    # Sort intersections by candidate index
    candidates = []
    for i in range(16):
        candidates.append([])
    
    for i in range(intersections.shape[0]):
        candidates[atom_indices[i]].append(intersections[i])
    
    # Construct convex hulls for each candidate
    candidate_clouds = []
    convex_hulls = []
    for i in range(16):
        if (len(candidates[i]) > 3):
            color = index_to_color(i)
            cloud = trimesh.points.PointCloud(candidates[i], colors=color)
            candidate_clouds.append(cloud)

            color.append(255)

            hull = cloud.convex_hull
            hull.visual.face_colors[:] = np.array(color)
            convex_hulls.append(hull)
    
    scene = trimesh.Scene(candidate_clouds)
    scene.show()

    scene = trimesh.Scene(convex_hulls)
    scene.show()

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
    elif args.Algorithm == 'convex_hull':
        visualize_convex_hull(args.Filename)
