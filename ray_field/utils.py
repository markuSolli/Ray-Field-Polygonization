import torch
import numpy as np
import open3d as o3d

from numpy import ndarray
from open3d.geometry import TriangleMesh
from open3d.utility import VerbosityLevel
from sklearn.neighbors import BallTree

def spherical_to_cartesian(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    """
    Args:
    - r: Radius
    - theta: Polar angle 
    - phi: Azimuthal angle

    Returns:
        - cartesian coordinate (tuple[x, y, z])
    """
    x: float = r * np.sin(theta) * np.cos(phi)
    y: float = r * np.sin(theta) * np.sin(phi)
    z: float = r * np.cos(theta)

    return (x, y, z)

# How to generate equidistributed points on the surface of a sphere
# https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
def generate_equidistant_sphere_points(N: int, r: float = 1.0) -> ndarray:
    """
    Args:
    - N: Number of points
    - r: Radius

    Returns:
    - points (ndarray[n, 3])
    """
    a: float = (4 * np.pi * r * 2) / N
    d: float = np.sqrt(a)
    m_theta: int = round(np.pi / d)
    d_theta: float = np.pi / m_theta
    d_phi: float = a / d_theta
    points: list[tuple] = []

    for m in range(m_theta):
        theta: float = np.pi * (m + 0.5) / m_theta
        m_phi: int = round(2 * np.pi * np.sin(theta) / d_phi)

        for n in range(m_phi):
            phi = (2 * np.pi * n) / m_phi
            points.append(spherical_to_cartesian(r, theta, phi))
    
    return np.array(points)

def normalize(vector: ndarray) -> ndarray:
    """
    Args:
    - vector (ndarray[3])
    
    Returns:
    - ndarray[3]
    """
    norm: float = np.linalg.norm(vector)

    if (norm == 0):
        return vector
    else:
        return vector / norm

def generate_rays_between_points(points: ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
    - points (ndarray[n, 3])

    Returns:
    - origins (Tensor[n, 3], dtype=float32)
    - directions (Tensor[n, 1, n-1, 3], dtype=float32)
    """
    dirs: list = []

    for i in range(points.shape[0]):
        dirs.append([[]])

        for j in range(i):
            direction: ndarray = normalize(points[j] - points[i])
            dirs[i][0].append(direction)
        
        for j in range(i + 1, points.shape[0]):
            direction: ndarray = normalize(points[j] - points[i])
            dirs[i][0].append(direction)
    
    origins = torch.Tensor(points).to(torch.float32)
    dirs = torch.Tensor(np.array(dirs)).to(torch.float32)

    return origins, dirs

def generate_rays_between_sphere_points(N: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
    - N (ndarray[n, 3])

    Returns:
    - origins (Tensor[n, 3], dtype=float32)
    - directions (Tensor[n, 1, n-1, 3], dtype=float32)
    """
    sphere_points: ndarray = generate_equidistant_sphere_points(N, 1.0)
    
    return generate_rays_between_points(sphere_points)

def poisson_surface_reconstruction(points: ndarray, normals: ndarray, depth: int, verbosity: VerbosityLevel = VerbosityLevel.Error) -> TriangleMesh:
    """
    Args:
    - points (ndarray[n, 3])
    - normals (ndarray[n, 3])
    - depth: Depth parameter in Poisson Surface Reconstruction (int)
    - verbosity (VerbosityLevel)

    Returns:
    - mesh (TriangleMesh)
    """
    # Create Open3D point cloud with normals
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    # Run Poisson Surface Reconstruction
    with o3d.utility.VerbosityContextManager(verbosity) as cm:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)[0]

    # Fix normals and assign color
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

    return mesh

def chamfer_distance(a: ndarray, b: ndarray) -> float:
    """
    Args:
        a (ndarray[n, 3])
        b (ndarray[m, 3])
    
    Returns:
        float
    """
    tree_a: BallTree = BallTree(a)
    tree_b: BallTree = BallTree(b)

    dist_a = tree_a.query(b)[0]
    dist_b = tree_b.query(a)[0]

    return dist_a.mean() + dist_b.mean()

def nearest_neighbor_distance(points: ndarray) -> ndarray:
    ball_tree: BallTree = BallTree(points)

    return ball_tree.query(points, k=2)[0][:, 1]