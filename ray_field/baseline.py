import torch
from ray_field import utils, CheckpointName, POISSON_DEPTH
from open3d.geometry import TriangleMesh
from numpy import ndarray

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

def baseline_hit_rate(model_name: CheckpointName) -> list[float]:
    model, device = utils.init_model(model_name)

    hit_rates = []

    with torch.no_grad():
        for N in range(100, 1001, 100):
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
