import torch
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class CandidateSphere(Algorithm):
    prescan_n: int = 60

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model, device = utils.init_model(model_name)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
            intersections, atom_indices = CandidateSphere._broad_scan(model, origins, dirs)

            radii, centers = CandidateSphere._generate_candidate_spheres(intersections, atom_indices, device)
            origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device)
            intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
        
        return utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)

    def hit_rate(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        model, device = utils.init_model(model_name)

        hit_rates = []

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
            broad_intersections, broad_atom_indices = CandidateSphere._broad_scan(model, origins, dirs)

            init_rays_n = dirs.shape[0] * dirs.shape[2]

            for N in N_values:
                print(N, end='\t')

                radii, centers = CandidateSphere._generate_candidate_spheres(broad_intersections, broad_atom_indices, device)
                origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device)
                intersections, _ = CandidateSphere._targeted_scan(model, origins, dirs)

                rays_n = dirs.shape[0] * dirs.shape[2]

                hit_rate = intersections.shape[0] / (rays_n + init_rays_n)
                hit_rates.append(hit_rate)
                print(f'{hit_rate:.3f}')

                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()
        
        return hit_rates

    def chamfer(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        model, device = utils.init_model(model_name)

        distances = []

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
            broad_intersections, broad_atom_indices = CandidateSphere._broad_scan(model, origins, dirs)

            for N in N_values:
                print(N, end='\t')

                radii, centers = CandidateSphere._generate_candidate_spheres(broad_intersections, broad_atom_indices, device)
                origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device)
                intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
                distance = utils.chamfer_distance_to_stanford(model_name, mesh, CandidateSphere.chamfer_samples)

                distances.append(distance)
                print(f'{distance:.6f}')
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances

    def hausdorff(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        model, device = utils.init_model(model_name)

        distances = []

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
            broad_intersections, broad_atom_indices = CandidateSphere._broad_scan(model, origins, dirs)

            for N in N_values:
                print(N, end='\t')

                radii, centers = CandidateSphere._generate_candidate_spheres(broad_intersections, broad_atom_indices, device)
                origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device)
                intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
                distance = utils.hausdorff_distance_to_stanford(model_name, mesh)

                distances.append(distance)
                print(f'{distance:.6f}')
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances

    def time(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)

        times = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values))

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                for j in range(CandidateSphere.time_samples):
                    torch.cuda.synchronize()
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
                    intersections, atom_indices = CandidateSphere._broad_scan(model, origins, dirs)

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]

                    radii, centers = CandidateSphere._generate_candidate_spheres(intersections, atom_indices, device)
                    origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device)
                    intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)

                    _ = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
                    
                    torch.cuda.synchronize()
                    time = timer() - start_time

                    if j == 0:
                        R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])

                    times[i] = times[i] + time
                    print(f'{time:.5f}', end='\t')
                    torch.cuda.empty_cache()
                
                print()
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        times = times / CandidateSphere.time_samples

        return times, R_values

    def time_steps(model_name: CheckpointName, N_values: list[int]) -> list[list[float]]:
        """Measure execution time of the surface reconstruction divided in
            - Broad scan
            - Targeted scan
            - Surface reconstruction

        Args:
            model_name (CheckpointName): Valid model name
            N_values (list[int]): N-values to test

        Returns:
            list[list[float]]: (N, 3) execution times
        """
        model, device = utils.init_model(model_name)

        times = np.zeros((len(N_values), 3))

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]

                for _ in range(CandidateSphere.time_samples):
                    torch.cuda.synchronize()
                    broad_start = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
                    intersections, atom_indices = CandidateSphere._broad_scan(model, origins, dirs)

                    torch.cuda.synchronize()
                    broad_end = timer()

                    radii, centers = CandidateSphere._generate_candidate_spheres(intersections, atom_indices, device)
                    origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device)
                    intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)

                    torch.cuda.synchronize()
                    targeted_end = timer()

                    _ = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)

                    torch.cuda.synchronize()
                    reconstruct_end = timer()

                    broad_time = broad_end - broad_start
                    targeted_time = targeted_end - broad_end
                    reconstruct_time = reconstruct_end - targeted_end

                    times[i][0] = times[i][0] + broad_time
                    times[i][1] = times[i][1] + targeted_time
                    times[i][2] = times[i][2] + reconstruct_time
                    print(f'N: {N}\t{broad_time:.4f}\t{targeted_time:.4f}\t{reconstruct_time:.4f}')
                    torch.cuda.empty_cache()
            
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return times / CandidateSphere.time_samples

    @staticmethod
    def _broad_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a ray query on the MARF model to get intersections and atom indices.

        Args:
            model (IntersectionFieldAutoDecoderModel): Initialized MARF model
            origins (torch.Tensor): Origins of the rays (n, 3)
            dirs (torch.Tensor): Ray directions (n, 1, n-1, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                - Intersection points (m, 3)
                - Atom indices (m)
        """        
        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False, return_all_atoms = True)

        intersections = result[2]
        is_intersecting = result[4]
        atom_indices = result[7]

        is_intersecting = is_intersecting.flatten()
        intersections = intersections.flatten(end_dim=2)[is_intersecting]
        atom_indices = atom_indices.flatten()[is_intersecting]

        return intersections, atom_indices

    @staticmethod
    def _generate_candidate_spheres(intersections: torch.Tensor, atom_indices: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Find spheres containing the intersections for each candidate.
        The sphere center is the mean position of the intersections.
        The sphere radius is the distance from the candidate's centroid and it's furthest point.

        Args:
            intersections (torch.Tensor): Intersections from the broad scan (x, 3)
            atom_indices (torch.Tensor): Atom indices from the broad scan (x)
            device (str): The device to store tensors in

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                - Sphere radii (16)
                - Sphere centers (16, 3)
        """        
        candidate_centers = torch.zeros((16, 3), dtype=torch.float32, device=device)
        max_radii = torch.zeros(16, dtype=torch.float32, device=device)
        candidate_n = torch.zeros(16, dtype=torch.int32, device=device)

        # Find candidate centers
        candidate_centers.index_add_(0, atom_indices, intersections)
        candidate_n.index_add_(0, atom_indices, torch.ones_like(atom_indices, dtype=torch.int32, device=device))

        mask = candidate_n > 0
        candidate_centers[mask] /= candidate_n[mask].unsqueeze(-1)

        # Find candidate radii
        distances = torch.norm(intersections - candidate_centers[atom_indices], dim=1)
        max_radii.index_reduce_(0, atom_indices, distances, reduce="amax")

        return max_radii, candidate_centers

    @staticmethod
    def _generate_candidate_rays(radii: torch.Tensor, centers: torch.Tensor, N: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates n origins along the unit sphere. For each non-empty candidate sphere,
        M rays are generated in a cone that covers that sphere. M is such that the total number of
        rays per origin m is <= (n - 1)

        Args:
            radii (torch.Tensor): Sphere radii (16)
            centers (torch.Tensor): Sphere centers (16, 3)
            N (int): Number of origins, the resulting n is <= N
            device (str): The device to store tensors in

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                - Ray origins (n, 3)
                - Ray directions (n, 1, m, 3)
        """        
        origins = utils.generate_equidistant_sphere_points_tensor(N, device)
        N = origins.shape[0]

        valid_mask = radii > 0
        valid_candidates = valid_mask.sum().item()
        M = (N - 1) // valid_candidates

        radii = radii[valid_mask]
        centers = centers[valid_mask]

        # Compute basis for origin -> sphere center
        central_dirs = centers[None, :, :] - origins[:, None, :]
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).expand(N, valid_candidates, 3).clone()
        mask = torch.abs(central_dirs[:,:, 2]) >= 0.9
        up[mask] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

        right = torch.cross(central_dirs, up)
        right = right / torch.norm(right, dim=2, keepdim=True)
        up = torch.cross(right, central_dirs)

        # Compute max angles
        dists = torch.norm(central_dirs, dim=2)
        safe_ratio = torch.clamp(radii / dists, min=-1.0, max=1.0)
        max_angles = torch.asin(safe_ratio).unsqueeze(-1)

        # Generate random spherical coordinates within max_angles
        theta = torch.rand(N, valid_candidates, M, device=device) * (2 * torch.pi)
        cos_beta = torch.rand(N, valid_candidates, M, device=device) * (1 - torch.cos(max_angles)) + torch.cos(max_angles)
        sin_beta = torch.sqrt(1 - cos_beta**2)

        directions = (
            cos_beta.unsqueeze(-1) * central_dirs.unsqueeze(2) +
            sin_beta.unsqueeze(-1) * torch.cos(theta).unsqueeze(-1) * right.unsqueeze(2) +
            sin_beta.unsqueeze(-1) * torch.sin(theta).unsqueeze(-1) * up.unsqueeze(2)
        ).reshape(N, 1, -1, 3)

        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        return origins.to(torch.float32), directions.to(torch.float32)

    
    @staticmethod
    def _targeted_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[ndarray, ndarray]:
        """Perform a ray query on the MARF model to get intersections and normals.
        The result will be transferred to the CPU.

        Args:
            model (IntersectionFieldAutoDecoderModel): Initialized MARF model
            origins (torch.Tensor): Origins of the rays (n, 3)
            dirs (torch.Tensor): Ray directions (n, 1, n-1, 3)

        Returns:
            tuple[ndarray, ndarray]
                - Intersection points (m, 3)
                - Intersection normals (m, 3)
        """
        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

        intersections = result[2]
        intersection_normals = result[3]
        is_intersecting = result[4]

        is_intersecting = is_intersecting.flatten()
        intersections = intersections.flatten(end_dim=2)[is_intersecting].cpu().detach().numpy()
        intersection_normals = intersection_normals.flatten(end_dim=2)[is_intersecting].cpu().detach().numpy()

        return intersections, intersection_normals
