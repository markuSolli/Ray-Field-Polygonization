import torch
import trimesh
import numpy as np
import torch.nn.functional as F

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import laptop_utils

def marf_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

    intersections = result[2]
    intersection_normals = result[3]
    is_intersecting = result[4]

    intersections = intersections.flatten(end_dim=2)
    intersection_normals = intersection_normals.flatten(end_dim=2)

    return intersections, intersection_normals

with torch.no_grad():
    model, device = laptop_utils.init_model('bunny')

    sphere = trimesh.primitives.Sphere(1, (0.0, 0.0, 0.0), None, 1)
    vertices = torch.from_numpy(sphere.vertices.copy()).to(torch.float32).to(device)

    dirs = F.normalize(-vertices).view(-1, 1, 1, 3)

    intersections, intersection_normals = marf_scan(model, vertices, dirs)

    sphere = trimesh.Trimesh(intersections, sphere.faces, vertex_normals=intersection_normals)

vertex_cloud = trimesh.points.PointCloud(vertices, colors=[0, 0, 0])
intersect_cloud = trimesh.points.PointCloud(intersections, colors=[255, 0, 0])

scene = trimesh.Scene([vertex_cloud, sphere])
scene.show()
