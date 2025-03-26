import torch
import numpy as np
import mesh_to_sdf
import trimesh

from numpy import ndarray
from trimesh import Trimesh
from sklearn.neighbors import BallTree
from ray_field import CheckpointName, get_checkpoint
from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ifield.data.stanford import read as stanford_read

chamfer_dict = {
    'bunny': 282,
    'happy_buddha': 318,
    'armadillo': 316,
    'dragon': 288,
}

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
    """Generates n points equally distributed along the unit sphere

    Args:
        N (int): Number of origins, the resulting n is <= N
        r (float, optional): Sphere radius. Defaults to 1.0.

    Returns:
        ndarray: The resulting points (n, 3)
    """
    a: float = (4 * np.pi * r**2) / N
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

# How to generate equidistributed points on the surface of a sphere
# https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
def generate_equidistant_sphere_points_tensor(N: int, device: str, r: float = 1.0) -> torch.Tensor:
    """Generates n points equally distributed along the unit sphere

    Args:
        N (int): Number of origins, the resulting n is <= N
        device (str): Which device to store the tensors in
        r (float, optional): Sphere radius. Defaults to 1.0.

    Returns:
        torch.Tensor: The resulting points (n, 3)
    """    
    a = (4 * np.pi * r**2) / N
    d = np.sqrt(a)
    m_theta = round(np.pi / d)
    d_theta = np.pi / m_theta
    d_phi = a / d_theta

    # Compute all theta and m_phi
    theta = torch.linspace(d_theta / 2, np.pi - d_theta / 2, m_theta, device=device)
    m_phi = torch.round(2 * np.pi * torch.sin(theta) / d_phi).long()

    # For each m_phi compute phi
    phi = [torch.linspace(0, (2 * np.pi * (m - 1)) / m, m, device=device) for m in m_phi]

    # Make theta and phi grid
    theta_expanded = torch.cat([torch.full((len(p),), t, device=device) for t, p in zip(theta, phi)])
    phi_expanded = torch.cat(phi)

    # Convert to cartesian
    x = r * torch.sin(theta_expanded) * torch.cos(phi_expanded)
    y = r * torch.sin(theta_expanded) * torch.sin(phi_expanded)
    z = r * torch.cos(theta_expanded)

    return torch.stack((x, y, z), dim=1)

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

def generate_rays_between_points(points: ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates ray directions for all pairs of points.

    Args:
        points (ndarray): Origin points (n, 3)
        device (str): Which device to store the tensors in

    Returns:
        tuple[torch.Tensor, torch.Tensor]
            - Ray origins, converted from the points array
            - Ray directions (n, 1, n-1, 3)
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
    
    origins = torch.from_numpy(points).to(torch.float32).to(device)
    dirs = torch.from_numpy(np.array(dirs)).to(torch.float32).to(device)

    return origins, dirs

def generate_rays_between_points_tensor(origins: torch.Tensor, device: str) -> torch.Tensor:
    """Generates ray directions for all pairs of origin points.

    Args:
        origins (torch.Tensor): Origin points (n, 3)
        device (str): Which device to store the tensors in

    Returns:
        torch.Tensor: Ray directions (n, 1, n-1, 3)
    """    
    N = origins.shape[0]
    
    # Compute all pairwise vectors and mask to exlude self
    vectors = origins.unsqueeze(0) - origins.unsqueeze(1)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    
    # Generate directions
    dirs = vectors[mask].reshape(N, N - 1, 3)    
    dirs = (dirs / torch.norm(dirs, dim=-1, keepdim=True)).unsqueeze(1)
    
    return dirs

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

def chamfer_distance(a: ndarray, b: ndarray) -> float:
    """Measure the Chamfer distance between two sets of points

    Args:
        a (ndarray): Set A (n, 3)
        b (ndarray): Set B (m, 3)

    Returns:
        float: The Chamfer distance
    """    
    tree_a: BallTree = BallTree(a)
    tree_b: BallTree = BallTree(b)

    dist_a = tree_a.query(b)[0]
    dist_b = tree_b.query(a)[0]

    return dist_a.mean() + dist_b.mean()

def hausdorff_distance(a: ndarray, b: ndarray) -> float:
    """Measures the Hausdorff distance between two sets of points

    Args:
        a (ndarray): Set A (n, 3)
        b (ndarray): Set B (m, 3)

    Returns:
        float: The Hausdorff distance
    """    
    tree_a: BallTree = BallTree(a)
    tree_b: BallTree = BallTree(b)

    dist_a = tree_a.query(b)[0]
    dist_b = tree_b.query(a)[0]

    return max(np.max(dist_a), np.max(dist_b))

def load_and_scale_stanford_mesh(model_name: CheckpointName) -> Trimesh:
    """Get the Stanford mesh for the model name, and scale it to the unit sphere.

    Args:
        model_name (CheckpointName): Valid model name of the Stanford mesh

    Returns:
        Trimesh: Scaled mesh
    """    
    stanford_mesh = stanford_read.read_mesh(model_name)
    return mesh_to_sdf.scale_to_unit_sphere(stanford_mesh)

def chamfer_distance_to_marf_1(model_name: CheckpointName) -> ndarray:
    model, device = init_model(model_name)
    N = chamfer_dict[model_name]

    origins, dirs = generate_sphere_rays_tensor(N, device)
    intersections = baseline_scan(model, origins, dirs)
    
    return intersections

def generate_rays_in_cone(points: torch.Tensor, angles: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    N = points.shape[0]
    M = N - 1

    # Central direction for each camera (opposite to the camera position)
    central_dirs = (-points).to(torch.float32)

    # Define an arbitrary "up" vector for each point (avoid collinearity)
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).expand(N, 3).clone()
    mask = torch.abs(central_dirs[:, 2]) >= 0.9
    up[mask] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

    # Compute right and up vectors for each frame
    right = torch.cross(central_dirs, up)
    right = right / torch.norm(right, dim=1, keepdim=True)
    up = torch.cross(right, central_dirs)

    # Generate random azimuth angles (theta) and cosine of tilt angles (beta)
    theta = torch.rand(N, M, device=device, dtype=torch.float32) * (2 * torch.pi)
    cos_beta = torch.rand(N, M, device=device, dtype=torch.float32) * (1 - torch.cos(angles)).unsqueeze(1) + torch.cos(angles).unsqueeze(1)
    sin_beta = torch.sqrt(1 - cos_beta**2)

    directions = (
        cos_beta.unsqueeze(-1) * central_dirs.unsqueeze(1) +
        sin_beta.unsqueeze(-1) * torch.cos(theta).unsqueeze(-1) * right.unsqueeze(1) +
        sin_beta.unsqueeze(-1) * torch.sin(theta).unsqueeze(-1) * up.unsqueeze(1)
    ).unsqueeze(1)

    return points.to(torch.float32), directions.to(torch.float32)

def init_model(model_name: CheckpointName) -> tuple[IntersectionFieldAutoDecoderModel, str]:
    """Initializes the MARF model, storing it on the device and turning on eval-mode.

    Args:
        model_name (CheckpointName): Valid model name

    Returns:
        tuple[IntersectionFieldAutoDecoderModel, str]
            - The MARF model
            - The device
    """    
    checkpoint = get_checkpoint(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = IntersectionFieldAutoDecoderModel.load_from_checkpoint(checkpoint).to(device)
    model.eval()

    return model, device

def generate_sphere_rays(N: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rays between all pairs of sphere points for the given N

    Args:
        N (int): Number of origins, the resulting n is <= N
        device (str): The device to store tensors in

    Returns:
        tuple[torch.Tensor, torch.Tensor]
            - Origin points (n, 3)
            - Ray directions (n, 1, n-1, 3)
    """
    sphere_points = generate_equidistant_sphere_points(N)
    
    return generate_rays_between_points(sphere_points, device)

def generate_sphere_rays_tensor(N: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rays between all pairs of sphere points for the given N

    Args:
        N (int): Number of origins, the resulting n is <= N
        device (str): The device to store tensors in

    Returns:
        tuple[torch.Tensor, torch.Tensor]
            - Origin points (n, 3)
            - Ray directions (n, 1, n-1, 3)
    """    
    origins = generate_equidistant_sphere_points_tensor(N, device)
    dirs = generate_rays_between_points_tensor(origins, device)
    
    return origins, dirs

def get_max_cone_angles(origins: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Calculate the maximum cone angles between the origins and the points.
    For each origin, the angles between the origin -> (0, 0, 0) and origin -> points are measured. 

    Args:
        origins (torch.Tensor): The origins (n, 3)
        points (torch.Tensor): The points (m, 3)

    Returns:
        torch.Tensor: The maximum angle for all origins (n)
    """    
    cam_forwards = -origins / torch.norm(origins, dim=1, keepdim=True)

    # Expand dimensions to align for broadcasting
    points_exp = points.unsqueeze(0)
    cam_pos_exp = origins.unsqueeze(1)
    cam_forward_exp = cam_forwards.unsqueeze(1)

    # Compute vectors angles
    vecs_to_points = points_exp - cam_pos_exp

    dot_products = torch.sum(vecs_to_points * cam_forward_exp, dim=2)
    vec_norms = torch.norm(vecs_to_points, dim=2)
    cam_norms = torch.norm(cam_forwards, dim=1, keepdim=True)

    cos_theta = dot_products / (vec_norms * cam_norms)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angles = torch.acos(cos_theta)
    return torch.max(angles, dim=1).values  

def baseline_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[ndarray, ndarray]:
    """Perform a ray query on the MARF model to get intersections.
    The result will be transferred to the CPU.

    Args:
        model (IntersectionFieldAutoDecoderModel): Initialized MARF model
        origins (torch.Tensor): Origins of the rays (n, 3)
        dirs (torch.Tensor): Ray directions (n, 1, n-1, 3)

    Returns:
        ndarray: Intersection points (m, 3)
    """        
    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = True)
    
    intersections = result[0]
    is_intersecting = result[1]

    is_intersecting = is_intersecting.flatten()
    intersections = intersections.flatten(end_dim=2)[is_intersecting].cpu().detach().numpy()
    
    return intersections