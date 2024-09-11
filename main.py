import trimesh
import numpy as np

from trimesh import Geometry, Scene
from numpy import ndarray, float64

def spherical_to_cartesian(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    """
    Args:
        r - Radius
        theta - Polar angle 
        phi - Azimuthal angle

    Returns:
        tuple[x, y, z]
    """
    x: float = r * np.sin(theta) * np.cos(phi)
    y: float = r * np.sin(theta) * np.sin(phi)
    z: float = r * np.cos(theta)

    return (x, y, z)

# How to generate equidistributed points on the surface of a sphere
# https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
def generate_equidistant_sphere_points(N: int, r: float = 1.0) -> np.ndarray:
    """
    Args:
        N - Number of points
        r - Radius

    Returns:
        ndarray[n, 3]
    """
    a: float = (4 * np.pi * r * 2) / N
    d: float = np.sqrt(a)
    m_theta: int = round(np.pi / d)
    d_theta: float = np.pi / m_theta
    d_phi: float = a / d_theta
    points = []

    for m in range(m_theta):
        theta: float = np.pi * (m + 0.5) / m_theta
        m_phi: int = round(2 * np.pi * np.sin(theta) / d_phi)

        for n in range(m_phi):
            phi = (2 * np.pi * n) / m_phi
            points.append(spherical_to_cartesian(r, theta, phi))
    
    return np.array(points)

def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if (norm == 0):
        return vector
    else:
        return vector / norm

def generate_rays_between_points(points: np.ndarray) -> np.ndarray:
    """
    Args:
        points (ndarray[n, 3])

    Returns:
        ndarray[n, 2, 3] - For dimension 1, index 0 is ray origin and index 1 is ray direction.
    """
    rays = []
    for i in range(points.shape[0]):
        for j in range(i):
            direction = normalize(points[j] - points[i])
            rays.append([points[i], direction])
        
        for j in range(i + 1, points.shape[0]):
            direction = normalize(points[j] - points[i])
            rays.append([points[i], direction])
    
    return np.array(rays)

mesh: Geometry = trimesh.load('suzanne.obj')
scale: ndarray[float64] = mesh.extents
transform: ndarray[float64] = trimesh.transformations.scale_matrix(2 / np.max(scale))
mesh.apply_transform(transform)

sphere_points = generate_equidistant_sphere_points(100, 1.0)
point_cloud = trimesh.points.PointCloud(sphere_points, colors=[0, 0, 255])

rays = generate_rays_between_points(sphere_points)

paths = []

for i in range(sphere_points.shape[0] - 1):
    paths.append(trimesh.load_path([rays[i, 0], rays[i, 0] + rays[i, 1]]))

scene: Scene = trimesh.Scene([mesh, point_cloud, paths])
scene.show()