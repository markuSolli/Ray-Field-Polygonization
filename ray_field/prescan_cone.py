import torch
from ray_field import utils, CheckpointName, get_checkpoint, POISSON_DEPTH
from ifield.models import intersection_fields
from open3d.geometry import TriangleMesh
from numpy import ndarray

PRESCAN_N = 100

def init_model(model_name: CheckpointName) -> tuple[intersection_fields.IntersectionFieldAutoDecoderModel, str]:
    checkpoint = get_checkpoint(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(checkpoint).to(device)
    model.eval()

    return model, device

def generate_sphere_rays(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    origins, dirs = utils.generate_rays_between_sphere_points(PRESCAN_N)
    origins = origins.to(device)
    dirs = dirs.to(device)

    return origins, dirs

def generate_cone_rays(intersections: torch.Tensor, N: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    r = intersections.abs().max().item()
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
    model, device = init_model(model_name)

    with torch.no_grad():
        origins, dirs = generate_sphere_rays(device)
        intersections = prescan_cone_broad_scan(model, device)

        origins, dirs = generate_cone_rays(intersections, N, device)
        intersections, intersection_normals = prescan_cone_targeted_scan(model, origins, dirs)
    
    return utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)

def prescan_cone_hit_rate(model_name: CheckpointName) -> list[float]:
    model, device = init_model(model_name)

    hit_rates = []

    with torch.no_grad():
        origins, dirs = generate_sphere_rays(device)
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