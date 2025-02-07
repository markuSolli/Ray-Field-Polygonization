import torch
from ray_field import utils, CheckpointName, POISSON_DEPTH
from open3d.geometry import TriangleMesh
from numpy import ndarray
from timeit import default_timer as timer

def baseline_scan(model, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[ndarray, ndarray]:
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

    intersections = result[2]
    intersection_normals = result[3]
    is_intersecting = result[4]

    is_intersecting = torch.flatten(is_intersecting)
    intersections = torch.flatten(intersections, end_dim=2)[is_intersecting].cpu().detach().numpy()
    intersection_normals = torch.flatten(intersection_normals, end_dim=2)[is_intersecting].cpu().detach().numpy()

    return intersections, intersection_normals

def baseline(model_name: CheckpointName, N: int) -> TriangleMesh:
    model, device = utils.init_model(model_name)
    origins, dirs = utils.generate_sphere_rays(device, N)

    with torch.no_grad():
        intersections, intersection_normals = baseline_scan(model, origins, dirs)

    return utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)

def baseline_hit_rate(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    hit_rates = []

    with torch.no_grad():
        for N in N_values:
            print(N, end='\t')
            origins, dirs = utils.generate_sphere_rays(device, N)
            sphere_n = origins.shape[0]
            rays_n = sphere_n * (sphere_n - 1)

            intersections, _ = baseline_scan(model, origins, dirs)
            
            hit_rate = intersections.shape[0] / rays_n
            hit_rates.append(hit_rate)
            print(f'{hit_rate:.3f}')

            torch.cuda.empty_cache()

    return hit_rates

def baseline_chamfer(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    distances = []

    with torch.no_grad():
        for N in N_values:
            print(N, end='\t')

            origins, dirs = utils.generate_sphere_rays(device, N)
            intersections, intersection_normals = baseline_scan(model, origins, dirs)
            mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
            distance = utils.chamfer_distance_to_stanford(model_name, mesh)

            distances.append(distance)
            print(f'{distance:.6f}')
            torch.cuda.empty_cache()

    return distances

def baseline_hausdorff(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    distances = []

    with torch.no_grad():
        for N in N_values:
            print(N, end='\t')

            origins, dirs = utils.generate_sphere_rays(device, N)
            intersections, intersection_normals = baseline_scan(model, origins, dirs)
            mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
            distance = utils.hausdorff_distance_to_stanford(model_name, mesh)

            distances.append(distance)
            print(f'{distance:.6f}')
            torch.cuda.empty_cache()

    return distances

def baseline_time(model_name: CheckpointName, N_values: list[int]) -> list[float]:
    model, device = utils.init_model(model_name)

    times = []

    with torch.no_grad():
        for N in N_values:
            print(N, end='\t')

            start_time = timer()

            origins, dirs = utils.generate_sphere_rays(device, N)
            intersections, intersection_normals = baseline_scan(model, origins, dirs)
            utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
            time = timer() - start_time

            times.append(time)
            print(f'{time:.6f}')
            torch.cuda.empty_cache()

    return times
