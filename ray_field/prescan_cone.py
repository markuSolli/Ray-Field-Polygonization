import torch
from ray_field import utils, CheckpointName, POISSON_DEPTH
from open3d.geometry import TriangleMesh
from numpy import ndarray

PRESCAN_N = 100

def generate_cone_rays(intersections: torch.Tensor, N: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    r = torch.norm(intersections, dim=1).max().item()
    origins, dirs = utils.generate_cone_rays_between_sphere_points(N, r)
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

def prescan_cone_hit_rate(model_name: CheckpointName) -> list[float]:
    model, device = utils.init_model(model_name)

    hit_rates = []

    with torch.no_grad():
        origins, dirs = utils.generate_sphere_rays(device, PRESCAN_N)
        broad_intersections = prescan_cone_broad_scan(model, origins, dirs)

        init_sphere_n = origins.shape[0]
        init_rays_n = init_sphere_n * (init_sphere_n - 1)

        for N in range(100, 1001, 100):
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