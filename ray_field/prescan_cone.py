import torch
import numpy as np
from ray_field import utils, CheckpointName, POISSON_DEPTH
from open3d.geometry import TriangleMesh
from numpy import ndarray

PRESCAN_N = 100

def generate_cone_rays(intersections: torch.Tensor, N: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    sphere_points = utils.generate_equidistant_sphere_points(N)
    intersections_cpu = intersections.cpu().clone()
    n = sphere_points.shape[0]
    angles = np.zeros(n)

    for i in range(n):
        # Compute basis
        z_axis = -sphere_points[i]
        x_candidate = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
        x_axis = x_candidate - np.dot(x_candidate, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Transform point cloud
        rot_matrix = np.stack([x_axis, y_axis, z_axis])
        transformed_points = torch.from_numpy(rot_matrix.astype(np.float32)) @ intersections_cpu.T

        # Project to xz-plane
        xz_projected = transformed_points[[0, 2], :].T

        # Compute cone angle
        r_max = torch.norm(xz_projected, dim=1).max().item()
        angles[i] = np.arctan(r_max)

    origins, dirs = utils.generate_rays_in_cone(sphere_points, angles)
    origins = origins.to(device)
    dirs = dirs.to(device)

    return origins, dirs

def prescan_cone_broad_scan(model, origins, dirs) -> torch.Tensor:
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)
    intersections = result[2]
    is_intersecting = result[4]

    is_intersecting = is_intersecting.flatten()
    return intersections.flatten(end_dim=2)[is_intersecting]

def prescan_cone_targeted_scan(model, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[ndarray, ndarray]:
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

    intersections = result[2]
    intersection_normals = result[3]
    is_intersecting = result[4]

    is_intersecting = is_intersecting.flatten()
    intersections = intersections.flatten(end_dim=2)[is_intersecting].cpu().detach().numpy()
    intersection_normals = intersection_normals.flatten(end_dim=2)[is_intersecting].cpu().detach().numpy()

    return intersections, intersection_normals

def prescan_cone(model_name: CheckpointName, N: int) -> TriangleMesh:
    model, device = utils.init_model(model_name)

    with torch.no_grad():
        origins, dirs = utils.generate_sphere_rays(device, PRESCAN_N)
        intersections = prescan_cone_broad_scan(model, origins, dirs)

        origins, dirs = generate_cone_rays(intersections, N, device)
        intersections, intersection_normals = prescan_cone_targeted_scan(model, origins, dirs)
    
    return utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)

def prescan_cone_hit_rate(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    hit_rates = []

    with torch.no_grad():
        origins, dirs = utils.generate_sphere_rays(device, PRESCAN_N)
        broad_intersections = prescan_cone_broad_scan(model, origins, dirs)

        init_sphere_n = origins.shape[0]
        init_rays_n = init_sphere_n * (init_sphere_n - 1)

        for N in N_values:
            print(N, end='\t')

            origins, dirs = generate_cone_rays(broad_intersections, N, device)
            intersections, _ = prescan_cone_targeted_scan(model, origins, dirs)

            sphere_n = origins.shape[0]
            rays_n = sphere_n * (sphere_n - 1)

            hit_rate = intersections.shape[0] / (rays_n + init_rays_n)
            hit_rates.append(hit_rate)
            print(f'{hit_rate:.3f}')

            torch.cuda.empty_cache()
    
    return hit_rates

def prescan_cone_chamfer(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    distances = []

    with torch.no_grad():
        for N in N_values:
            print(N, end='\t')

            origins, dirs = utils.generate_sphere_rays(device, PRESCAN_N)
            intersections = prescan_cone_broad_scan(model, origins, dirs)

            origins, dirs = generate_cone_rays(intersections, N, device)
            intersections, intersection_normals = prescan_cone_targeted_scan(model, origins, dirs)
    
            mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
            distance = utils.chamfer_distance_to_stanford(model_name, mesh)

            distances.append(distance)
            print(f'{distance:.6f}')
            torch.cuda.empty_cache()

    return distances

def prescan_cone_hausdorff(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    distances = []

    with torch.no_grad():
        for N in N_values:
            print(N, end='\t')

            origins, dirs = utils.generate_sphere_rays(device, PRESCAN_N)
            intersections = prescan_cone_broad_scan(model, origins, dirs)

            origins, dirs = generate_cone_rays(intersections, N, device)
            intersections, intersection_normals = prescan_cone_targeted_scan(model, origins, dirs)
    
            mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
            distance = utils.hausdorff_distance_to_stanford(model_name, mesh)

            distances.append(distance)
            print(f'{distance:.6f}')
            torch.cuda.empty_cache()

    return distances

def prescan_cone_optimize(model_name: CheckpointName, N_values: list[int], M_values: list[int]) -> list[list[float]]:
    model, device = utils.init_model(model_name)

    distances = []

    with torch.no_grad():
        for M in M_values:
            origins, dirs = utils.generate_sphere_rays(device, M)
            broad_intersections = prescan_cone_broad_scan(model, origins, dirs)
            prescan_distances = []

            for N in N_values:
                print(f'M: {M}\tN: {N}', end='\t')

                origins, dirs = generate_cone_rays(broad_intersections, N, device)
                intersections, intersection_normals = prescan_cone_targeted_scan(model, origins, dirs)
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
                distance = utils.chamfer_distance_to_stanford(model_name, mesh)

                prescan_distances.append(distance)
                print(f'{distance:.6f}')
                torch.cuda.empty_cache()
            
            distances.append(prescan_distances)
            torch.cuda.empty_cache()

    return distances