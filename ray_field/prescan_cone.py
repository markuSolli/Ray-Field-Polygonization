import torch
from ray_field import utils, CheckpointName, POISSON_DEPTH
from open3d.geometry import TriangleMesh
from numpy import ndarray
from timeit import default_timer as timer

PRESCAN_N = 100

def generate_cone_rays(intersections: torch.Tensor, N: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    sphere_points = torch.from_numpy(utils.generate_equidistant_sphere_points(N)).to(device)
    cam_forwards = -sphere_points / torch.norm(sphere_points, dim=1, keepdim=True)

    # Expand dimensions to align for broadcasting
    intersections_exp = intersections.unsqueeze(0)
    cam_pos_exp = sphere_points.unsqueeze(1)
    cam_forward_exp = cam_forwards.unsqueeze(1)

    # Compute vectors angles
    vecs_to_points = intersections_exp - cam_pos_exp

    dot_products = torch.sum(vecs_to_points * cam_forward_exp, dim=2)
    vec_norms = torch.norm(vecs_to_points, dim=2)
    cam_norms = torch.norm(cam_forwards, dim=1, keepdim=True)

    cos_theta = dot_products / (vec_norms * cam_norms)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angles = torch.acos(cos_theta)
    max_angles = torch.max(angles, dim=1).values

    # Generate rays
    origins, dirs = utils.generate_rays_in_cone(sphere_points, max_angles, device)
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

def prescan_cone_time(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    times = []

    with torch.no_grad():
        for N in N_values:
            print(N, end='\t')

            start_time = timer()

            origins, dirs = utils.generate_sphere_rays(device, PRESCAN_N)
            intersections = prescan_cone_broad_scan(model, origins, dirs)

            origins, dirs = generate_cone_rays(intersections, N, device)
            intersections, intersection_normals = prescan_cone_targeted_scan(model, origins, dirs)
            utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
            time = timer() - start_time

            times.append(time)
            print(f'{time:.6f}')
            torch.cuda.empty_cache()

    return times

def prescan_cone_time_steps(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    times = []

    with torch.no_grad():
        for N in N_values:
            print(N, end='\t')

            broad_start = timer()
            origins, dirs = utils.generate_sphere_rays(device, PRESCAN_N)
            intersections = prescan_cone_broad_scan(model, origins, dirs)
            broad_end = timer()

            origins, dirs = generate_cone_rays(intersections, N, device)
            intersections, intersection_normals = prescan_cone_targeted_scan(model, origins, dirs)
            targeted_end = timer()

            utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
            reconstruct_end = timer()

            broad_time = broad_end - broad_start
            targeted_time = targeted_end - broad_end
            reconstruct_time = reconstruct_end - targeted_end

            times.append([broad_time, targeted_time, reconstruct_time])
            print(f'{broad_time:.4f}\t{targeted_time:.4f}\t{reconstruct_time:.4f}')
            torch.cuda.empty_cache()

    return times