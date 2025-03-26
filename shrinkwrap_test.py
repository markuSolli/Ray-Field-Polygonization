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
    # Init model and sphere
    model, device = laptop_utils.init_model('bunny')
    mesh = trimesh.primitives.Sphere(1, (0.0, 0.0, 0.0), None, 2)

    scene = trimesh.Scene([mesh])
    scene.show(flags={'wireframe': True})

    # Generate rays inwards
    origins = torch.from_numpy(mesh.vertices.copy()).to(torch.float32).to(device)
    dirs = F.normalize(-origins).view(-1, 1, 1, 3)

    origin_cloud = trimesh.points.PointCloud(origins, colors=[0, 0, 255])
    paths = []
    for i in range(dirs.shape[0]):
        paths.append(trimesh.load_path([origins[i], origins[i] + dirs[i,0,0] * 0.2]))
    
    scene = trimesh.Scene([mesh, origin_cloud, paths])
    scene.show(flags={'wireframe': True})

    # Query MARF
    intersections, intersection_normals, is_intersecting = marf_scan(model, origins, dirs)

    intersect_cloud = trimesh.points.PointCloud(intersections, colors=[255, 0, 0])
    scene = trimesh.Scene([mesh, origin_cloud, paths, intersect_cloud])
    scene.show(flags={'wireframe': True})

    # Shrink vertices onto model
    mesh = trimesh.Trimesh(intersections, mesh.faces, vertex_normals=intersection_normals)

    scene = trimesh.Scene([mesh])
    scene.show()

    for iteration in range(4):
        # Generate rays from face centers
        faces = mesh.faces.copy()
        vertices = torch.from_numpy(mesh.vertices.copy()).to(torch.float32).to(device)

        origins = torch.zeros((faces.shape[0], 3), dtype=torch.float32)
        dirs = torch.zeros((faces.shape[0], 1, 1, 3), dtype=torch.float32)

        for i in range(faces.shape[0]):
            face = faces[i]
            
            center = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0

            normal = torch.cross(vertices[face[2]] - vertices[face[0]], vertices[face[1]] - vertices[face[0]])
            normal = normal / torch.norm(normal)

            origins[i] = center
            dirs[i][0][0] = normal

        origin_cloud = trimesh.points.PointCloud(origins, colors=[0, 0, 255])
        paths = []
        for i in range(dirs.shape[0]):
            paths.append(trimesh.load_path([origins[i], origins[i] + dirs[i,0,0] * 0.1]))
        
        #scene = trimesh.Scene([mesh, origin_cloud, paths])
        #scene.show(flags={'wireframe': True})

        # Query MARF
        intersections, intersection_normals, is_intersecting = marf_scan(model, origins, dirs)

        #intersect_cloud = trimesh.points.PointCloud(intersections, colors=[255, 0, 0])
        #scene = trimesh.Scene([mesh, intersect_cloud])
        #scene.show(flags={'wireframe': True})

        # Filter near and far intersections
        far_faces = []
        near_faces = []
        for i in range(intersections.shape[0]):
            if (is_intersecting[i] and torch.norm(intersections[i] - origins[i]) > 0.01):
                far_faces.append(i)
            else:
                near_faces.append(i)
        
        far_intersections = []
        near_intersections = []

        for index in far_faces:
            far_intersections.append(intersections[index])
        for index in near_faces:
            near_intersections.append(intersections[index])
        
        #far_cloud = trimesh.points.PointCloud(far_intersections, colors=[255, 0, 0])
        #near_cloud = trimesh.points.PointCloud(near_intersections, colors=[0, 0, 255])
        #scene = trimesh.Scene([mesh, far_cloud, near_cloud])
        #scene.show(flags={'wireframe': True})

        # Split faces
        vertices = mesh.vertices.copy().tolist()
        normals = mesh.vertex_normals.copy().tolist()
        faces = faces.tolist()
        v_index = len(vertices)

        new_faces = []
        for index in near_faces:
            new_faces.append(faces[index])

        for i in range(len(far_faces)):
            face_index = far_faces[i]
            face_vertices = faces[face_index]

            new_faces.append([face_vertices[0], face_vertices[1], v_index])
            new_faces.append([face_vertices[1], face_vertices[2], v_index])
            new_faces.append([face_vertices[2], face_vertices[0], v_index])

            vertices.append(intersections[face_index])
            normals.append(intersection_normals[face_index])
            v_index = v_index + 1
        
        mesh = trimesh.Trimesh(vertices, new_faces, vertex_normals=normals)

        scene = trimesh.Scene([mesh])
        scene.show()
