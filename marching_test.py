import torch
import trimesh
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import laptop_utils
from numpy import ndarray
from scipy.spatial import KDTree

EPS = 0.05
DELTA = 0.04

def marf_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

    intersections = result[2]
    intersection_normals = result[3]
    is_intersecting = result[4]

    intersections = intersections.flatten(end_dim=2)
    intersection_normals = intersection_normals.flatten(end_dim=2)
    is_intersecting = torch.flatten(is_intersecting)

    return intersections, intersection_normals, is_intersecting

def construct_seed_triangle(model: IntersectionFieldAutoDecoderModel, device: str) -> tuple[ndarray, ndarray, list, list, list]:
    origins = torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32)

    dirs = torch.tensor([
        [0.0, 0.0, 1.0],
        [EPS, 0.0, 1.0],
        [0.0, EPS, 1.0],
    ], device=device, dtype=torch.float32)

    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    dirs = dirs.view(1, 1, -1, 3)

    intersections, intersection_normals, is_intersecting = marf_scan(model, origins, dirs)

    if not is_intersecting.all():
        print("One or more seed rays did not intersect.")
        exit()
    
    vertices = intersections.cpu().detach().numpy()
    normals = intersection_normals.cpu().detach().numpy()
    faces = [[0, 1, 2]]
    edges = [[0, 1], [1, 2], [2, 0]]
    edge_to_face = [0, 0, 0]

    return vertices, normals, faces, edges, edge_to_face

def visualize(vertices: list, normals: list, faces: list) -> None:
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    mesh.fix_normals()

    scene = trimesh.Scene([mesh])
    scene.show()

def visualize_x_proj(vertices: ndarray, normals: ndarray, faces: list, midpoint: list, x_proj: list) -> None:
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    mesh.fix_normals()

    midpoint_cloud = trimesh.points.PointCloud([midpoint], colors=[255, 0, 0])
    x_proj_cloud = trimesh.points.PointCloud([x_proj], colors=[0, 0, 255])

    scene = trimesh.Scene([mesh, midpoint_cloud, x_proj_cloud])
    scene.show()

def visualize_x_proj_evaluation(vertices: ndarray, normals: ndarray, faces: list, x_proj: list, face_normal: ndarray) -> None:
    origin = x_proj + (face_normal * 0.01)

    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    mesh.fix_normals()

    x_proj_cloud = trimesh.points.PointCloud([x_proj], colors=[0, 0, 255])
    origin_cloud = trimesh.points.PointCloud([origin], colors=[255, 0, 0])
    dir_path = trimesh.load_path([origin, origin - (face_normal * 0.01)])

    scene = trimesh.Scene([mesh, x_proj_cloud, origin_cloud, dir_path])
    scene.show()

def visualize_x_new(vertices: ndarray, normals: ndarray, faces: list, x_proj: list, x_new: ndarray, x_new_normal: ndarray) -> None:
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    mesh.fix_normals()

    x_proj_cloud = trimesh.points.PointCloud([x_proj], colors=[0, 0, 255])
    x_new_cloud = trimesh.points.PointCloud([x_new], colors=[255, 0, 0])
    normal_path = trimesh.load_path([x_new, x_new + (x_new_normal * 0.01)])

    scene = trimesh.Scene([mesh, x_proj_cloud, x_new_cloud, normal_path])
    scene.show()

def visualize_new_face(vertices: ndarray, normals: ndarray, faces: list, a, b, c) -> None:
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    mesh.fix_normals()

    face_path = trimesh.load_path([a, b, c, a])

    scene = trimesh.Scene([mesh, face_path])
    scene.show(flags={'wireframe': True})

def visualize_circumsphere(vertices: ndarray, normals: ndarray, faces: list, edge: list, x_new: ndarray, circumcenter: ndarray, radius: float) -> None:
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    mesh.fix_normals()

    x_new_cloud = trimesh.points.PointCloud([x_new], colors=[255, 0, 0])
    face_path = trimesh.load_path([vertices[edge[0]], x_new, vertices[edge[1]], vertices[edge[0]]])
    circumsphere = trimesh.primitives.Sphere(radius, circumcenter)
    circumsphere.visual.face_colors[:] = np.array([0, 0, 255, 64])

    scene = trimesh.Scene([mesh, x_new_cloud, face_path, circumsphere])
    scene.show()

def get_kd_query_indices(a, b, c, vertices, face_normal) -> list:
    mid_AB = (b + a) / 2.0 
    mid_AC = (c + a) / 2.0

    bisector_AB = np.cross(b - a, face_normal)
    bisector_AC = np.cross(c - a, face_normal)

    A_matrix = np.vstack([bisector_AB, -bisector_AC]).T
    b_vector = mid_AC - mid_AB

    try:
        t = np.linalg.lstsq(A_matrix, b_vector, rcond=None)[0]
        circumcenter = mid_AB + (t[0] * bisector_AB)
    except np.linalg.LinAlgError:
        print("Points are co-linear")
        return []

    radius = np.linalg.norm(a - circumcenter)
    tree = KDTree(vertices)
    return tree.query_ball_point(circumcenter, radius)

def delaunay_constraint(a, b_idx, c_idx, vertices, face_normal) -> tuple[bool, list]:
    kd_query_indices = get_kd_query_indices(a, vertices[b_idx], vertices[c_idx], vertices, face_normal)
    kd_query_indices = [idx for idx in kd_query_indices if idx not in {b_idx, c_idx}]
    
    if len(kd_query_indices) == 0:
        return True, []
    else:
        return False, kd_query_indices

def delaunay_constraint_existing(a_idx, b_idx, c_idx, vertices, face_normal) -> bool:
    kd_query_indices = get_kd_query_indices(vertices[a_idx], vertices[b_idx], vertices[c_idx], vertices, face_normal)
    kd_query_indices = [idx for idx in kd_query_indices if idx not in {a_idx, b_idx, c_idx}]
    
    return (len(kd_query_indices) == 0)

def marching_triangles(model: IntersectionFieldAutoDecoderModel, device: str, vertices: ndarray, normals: ndarray, faces: list, edges: list, edge_to_face: list):
    i = -1

    while ((i + 1) < len(edges)):
        i = i + 1
        edge = edges[i]
        face = faces[edge_to_face[i]]

        # Estimate x_proj
        midpoint = (vertices[edge[0]] + vertices[edge[1]]) / 2.0
        edge_direction = vertices[edge[1]] - vertices[edge[0]]
        
        face_normal = np.cross(vertices[face[2]] - vertices[face[0]], vertices[face[1]] - vertices[face[0]])
        face_normal = face_normal / np.linalg.norm(face_normal)

        if np.dot(face_normal, normals[face[0]]) < 0:
            face_normal = -face_normal

        perpendicular = np.cross(edge_direction, face_normal)
        perpendicular = perpendicular / np.linalg.norm(perpendicular)

        centroid = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
        
        if np.dot(centroid - midpoint, perpendicular) > 0:
            perpendicular = -perpendicular

        x_proj = midpoint + (perpendicular * DELTA)

        # Evaluate nearest point x_new
        origins = torch.tensor((x_proj + face_normal), device=device, dtype=torch.float32).view(-1, 3)
        dirs = torch.tensor((-face_normal), device=device, dtype=torch.float32).view(1, 1, -1, 3)

        intersections, intersection_normals, is_intersecting = marf_scan(model, origins, dirs)

        if not is_intersecting.all():
            print("Nearest point evaluation did not intersect.")
            continue
    
        x_new = intersections.cpu().detach().numpy()[0]
        x_new_normal = intersection_normals.cpu().detach().numpy()[0]

        # Terminate if surface orientation has flipped
        if np.dot(face_normal, x_new_normal) < 0:
            print("Terminating x_new due to opposite surface normal")
            continue

        # Apply 3D Delaunay surface constraint to T_new
        isConstraintPassed, query_indices = delaunay_constraint(x_new, edge[0], edge[1], vertices, face_normal)
        
        if isConstraintPassed:
            # Add T_new to M
            new_vertex_idx = vertices.shape[0]
            new_face_idx = len(faces)

            faces.append([edge[0], edge[1], new_vertex_idx])
            edges.append([edge[0], new_vertex_idx])
            edges.append([edge[1], new_vertex_idx])
            edge_to_face.append(new_face_idx)
            edge_to_face.append(new_face_idx)

            vertices = np.vstack((vertices, x_new))
            normals = np.vstack((normals, x_new_normal))
        else:
            for query_index in query_indices:
                if delaunay_constraint_existing(query_index, edge[0], edge[1], vertices, face_normal):
                    if ([query_index, edge[0]] not in edges) and ([edge[0], query_index] not in edges):
                        existing_vertex = edge[0]
                    elif ([query_index, edge[1]] not in edges) and ([edge[1], query_index] not in edges):
                        existing_vertex = edge[1]
                    else:
                        print("Both edges already exists")
                        continue

                    # Add T_new to M
                    new_face_idx = len(faces)

                    faces.append([edge[0], edge[1], query_index])
                    edges.append([existing_vertex, query_index])
                    edge_to_face.append(new_face_idx)

                    break
    
    visualize(vertices, normals, faces)

def main():
    model, device = laptop_utils.init_model('bunny')
    vertices, normals, faces, edges, edge_to_face = construct_seed_triangle(model, device)
    marching_triangles(model, device, vertices, normals, faces, edges, edge_to_face)

main()