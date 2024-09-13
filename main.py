import trimesh
import numpy as np

from trimesh import Trimesh, Scene, PointCloud
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
def generate_equidistant_sphere_points(N: int, r: float = 1.0) -> ndarray:
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
    points: list[tuple] = []

    for m in range(m_theta):
        theta: float = np.pi * (m + 0.5) / m_theta
        m_phi: int = round(2 * np.pi * np.sin(theta) / d_phi)

        for n in range(m_phi):
            phi = (2 * np.pi * n) / m_phi
            points.append(spherical_to_cartesian(r, theta, phi))
    
    return np.array(points)

def normalize(vector: ndarray) -> ndarray:
    norm: float = np.linalg.norm(vector)

    if (norm == 0):
        return vector
    else:
        return vector / norm

def generate_rays_between_points(points: ndarray) -> ndarray:
    """
    Args:
        points (ndarray[n, 3])

    Returns:
        ndarray[n, 2, 3] - For dimension 1, index 0 is ray origin and index 1 is ray direction.
    """
    rays: list = []

    for i in range(points.shape[0]):
        for j in range(i):
            direction: ndarray = normalize(points[j] - points[i])
            rays.append([points[i], direction])
        
        for j in range(i + 1, points.shape[0]):
            direction: ndarray = normalize(points[j] - points[i])
            rays.append([points[i], direction])
    
    return np.array(rays)

# Read mesh from file
mesh: Trimesh = trimesh.load_mesh('suzanne.obj')

# Scale down mesh to fit unit circle
scale: ndarray[float64] = mesh.extents
transform: ndarray[float64] = trimesh.transformations.scale_matrix(2 / np.max(scale))
mesh.apply_transform(transform)

# Generate points along the unit sphere
sphere_points: ndarray = generate_equidistant_sphere_points(100, 1.0)

# Generate rays between all points
rays: ndarray = generate_rays_between_points(sphere_points)

# Perform ray intersections on the mesh
locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=rays[:, 0], ray_directions=rays[:, 1])

# Get face normals for intersection points
normals: ndarray = mesh.face_normals[index_tri]

# Visualize normals
paths = []

for i in range(0, normals.shape[0], 10):
    paths.append(trimesh.load_path([locations[i], locations[i] + normals[i] / 10]))

# Create a point cloud of the intersection locations
point_cloud: PointCloud = PointCloud(locations, colors=[255, 0, 0])

# Visualize
scene: Scene = Scene([mesh, point_cloud, paths])
scene.show()