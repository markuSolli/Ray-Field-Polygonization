import torch
import trimesh
import numpy as np

from ray_field import utils
from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel

def baseline_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:      
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

    intersections = result[2].squeeze()
    intersection_normals = result[3].squeeze()
    is_intersecting = result[4].squeeze()

    return intersections, intersection_normals, is_intersecting

# Baseline scan
with torch.no_grad():
    model, device = utils.init_model('bunny')
    origins, dirs = utils.generate_sphere_rays_tensor(10, device)
    intersections, intersection_normals, is_intersecting = baseline_scan(model, origins, dirs)
    dirs = dirs.squeeze()

    cosine = torch.sum(dirs * intersection_normals, dim=-1)
    mask = (cosine < -0.707) & (cosine >= -1) & is_intersecting

    debug_intersections = intersections.clone()
    debug_normals = intersection_normals.clone()

    intersections = intersections[mask]
    intersection_normals = intersection_normals[mask]

    debug_intersections[~mask] = 0
    debug_normals[~mask] = 0

    intersections = intersections.cpu().detach().numpy()
    origins = origins.cpu().detach().numpy()
    dirs = dirs.cpu().detach().numpy()
    debug_intersections = debug_intersections.cpu().detach().numpy()
    debug_normals = debug_normals.cpu().detach().numpy()

    intersect_cloud = trimesh.points.PointCloud(intersections)
    origin_cloud = trimesh.points.PointCloud(origins, colors=[0, 255, 0])

    for i in range(debug_intersections.shape[0]):
        for j in range(debug_intersections.shape[1]):
            if np.all(debug_intersections[i, j] == 0) or np.all(debug_normals[i, j] == 0):
                continue

            ray_length = np.linalg.norm(origins[i] - debug_intersections[i, j])
            ray_path = trimesh.load_path([origins[i], origins[i] + dirs[i, j] * ray_length])
            ray_path.colors = [[255, 0, 0]] * len(ray_path.entities)
            normal_path = trimesh.load_path([debug_intersections[i, j], debug_intersections[i, j] + debug_normals[i, j] * 0.1])
            normal_path.colors = [[0, 0, 255]] * len(normal_path.entities)

            scene = trimesh.Scene([intersect_cloud, origin_cloud, ray_path, normal_path])
            scene.show()
