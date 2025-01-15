import numpy as np
import torch

from numpy import ndarray

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
    """
    Args:
        vector (ndarray[3])
    
    Returns:
        ndarray[3]
    """
    norm: float = np.linalg.norm(vector)

    if (norm == 0):
        return vector
    else:
        return vector / norm

def generate_rays_between_points(points: ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    origins: list = []
    dirs: list = []

    for i in range(points.shape[0]):
        origins.append(points[i])
        dirs.append([[]])

        for j in range(i):
            direction: ndarray = normalize(points[j] - points[i])
            dirs[i][0].append(direction)
        
        for j in range(i + 1, points.shape[0]):
            direction: ndarray = normalize(points[j] - points[i])
            dirs[i][0].append(direction)
    
    origins = torch.Tensor(np.array(origins)).to(torch.float32)
    dirs = torch.Tensor(np.array(dirs)).to(torch.float32)

    return origins, dirs

def generate_rays_between_sphere_points(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    sphere_points: ndarray = generate_equidistant_sphere_points(n, 1.0)
    
    return generate_rays_between_points(sphere_points)
