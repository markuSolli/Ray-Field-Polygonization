import torch
from ray_field import utils, CheckpointName, get_checkpoint, POISSON_DEPTH
from ifield.models import intersection_fields
from open3d.geometry import TriangleMesh

def baseline(model_name: CheckpointName, N: int) -> TriangleMesh:
    checkpoint = get_checkpoint(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(checkpoint)
    model.eval().to(device)

    origins, dirs = utils.generate_rays_between_sphere_points(N)
    origins = origins.to(device)
    dirs = dirs.to(device)

    with torch.no_grad():
        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

        intersections = result[2].cpu()
        intersection_normals = result[3].cpu()
        is_intersecting = result[4].cpu()

        is_intersecting = torch.flatten(is_intersecting)
        intersections = torch.flatten(intersections, end_dim=2)[is_intersecting].detach().numpy()
        intersection_normals = torch.flatten(intersection_normals, end_dim=2)[is_intersecting].detach().numpy()

    return utils.poisson_surface_reconstruction(intersections, intersection_normals, POISSON_DEPTH)
