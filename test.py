import torch
import trimesh
import numpy as np
import torch.nn.functional as F

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import laptop_utils

def marf_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

    intersections = result[2]
    intersection_normals = result[3]
    is_intersecting = result[4]

    intersections = intersections.flatten(end_dim=2)
    intersection_normals = intersection_normals.flatten(end_dim=2)
    is_intersecting = torch.flatten(is_intersecting)

    return intersections, intersection_normals, is_intersecting

with torch.no_grad():
    # Query seed triangle
    model, device = laptop_utils.init_model('bunny')
    origins = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32).view(-1, 3)

    eps = 0.05
    dirs = torch.tensor([
        [0.0, 0.0, 1.0],
        [eps, 0.0, 1.0],
        [0.0, eps, 1.0],
    ], device=device, dtype=torch.float32)

    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    dirs = dirs.view(1, 1, -1, 3)

    intersections, intersection_normals, is_intersecting = marf_scan(model, origins, dirs)

    if not is_intersecting.all():
        print("One or more rays did not intersect.")
        exit()

    # Construct seed triangle
    delta = 0.02
    faces = [[0, 1, 2]]
    vertices = intersections.detach().numpy().tolist()
    normals = intersection_normals.detach().numpy().tolist()
    face_centers = [(intersections[0] + intersections[1] + intersections[2]) / 3.0]
    normal = torch.cross(intersections[2] - intersections[0], intersections[1] - intersections[0])
    normal = normal / torch.norm(normal)
    face_normals = [normal]
    edge_to_face = [0, 0, 0]
    edges = [[0, 1], [1, 2], [2, 0]]

    # Construct candidate vertices
    origins = torch.zeros((len(edges), 3), dtype=torch.float32, device=device)
    dirs = torch.zeros((len(edges), 1, 1, 3), dtype=torch.float32, device=device)

    for i in range(len(edges)):
        edge = edges[i]
        face_idx = edge_to_face[i]

        edge_center = (intersections[edge[0]] + intersections[edge[1]]) / 2.0
        direction = edge_center - face_centers[face_idx]
        direction = direction / torch.norm(direction)
        new_vertex = face_centers[face_idx] + (direction * delta)

        origins[i] = new_vertex + face_normals[face_idx]
        dir = new_vertex - origins[i]
        dir = dir / torch.norm(dir)
        dirs[i][0][0] = dir

        continue

        paths = []

        normal_path = trimesh.load_path([face_centers[face_idx], face_centers[face_idx] + (face_normals[face_idx] * delta)])
        normal_path.colors = [[0, 0, 255]] * len(normal_path.entities)
        paths.append(normal_path)

        for j in range(len(edges)):
            edge_path = trimesh.load_path([intersections[edges[j][0]], intersections[edges[j][1]]])
            if (j == i):
                edge_path.colors = [[0, 0, 255]] * len(edge_path.entities)
            paths.append(edge_path)

        vertex_cloud = trimesh.points.PointCloud([new_vertex], colors=[255, 0, 0])

        scene = trimesh.Scene([paths, vertex_cloud])
        scene.show()
    
    # Query MARF
    intersections, intersection_normals, is_intersecting = marf_scan(model, origins, dirs)
    vertices_next_idx = len(vertices)
    faces_next_idx = len(faces)

    paths = []
    for j in range(len(edges)):
        paths.append(trimesh.load_path([vertices[edges[j][0]], vertices[edges[j][1]]]))

    intersect_cloud = trimesh.points.PointCloud(intersections, colors=[255, 0, 0])

    scene = trimesh.Scene([paths, intersect_cloud])
    scene.show()

    for i in range(intersections.shape[0]):
        if (not is_intersecting[i]):
            continue

        edge = edges[i]
        face_idx = edge_to_face[i]

        vertices.append(intersections[i])
        normals.append(intersection_normals[i])

        faces.append([edge[0], edge[1], vertices_next_idx])
        edges.append([edge[0], vertices_next_idx])
        edges.append([edge[1], vertices_next_idx])
        edge_to_face.append(faces_next_idx)
        edge_to_face.append(faces_next_idx)

        vertices_next_idx = vertices_next_idx + 1
        faces_next_idx = faces_next_idx + 1

    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    scene = trimesh.Scene([mesh])
    scene.show()
