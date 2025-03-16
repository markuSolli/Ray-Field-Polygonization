import torch
import trimesh
import numpy as np
import torch.nn.functional as F

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import laptop_utils
from numpy import ndarray

def index_to_color(index: int):
    r = (index % 4) * 64
    g = (index % 6) * 43
    b = (index % 8) * 32

    return [r, g, b]

def marf_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False, return_all_atoms = True)
 
    intersections = result[2]
    is_intersecting = result[4]
    atom_indices = result[7]
 
    is_intersecting = is_intersecting.flatten()
    intersections = intersections.flatten(end_dim=2)[is_intersecting]
    atom_indices = atom_indices.flatten()[is_intersecting]
 
    return intersections, atom_indices

with torch.no_grad():
    # Broad scan
    model, device = laptop_utils.init_model('bunny')
    origins, dirs = laptop_utils.generate_sphere_rays_tensor(100, device)
    intersections, atom_indices = marf_scan(model, origins, dirs)

    candidates = []
    for i in range(16):
        candidates.append([])
    
    for i in range(intersections.shape[0]):
        candidates[atom_indices[i]].append(intersections[i])
    
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
