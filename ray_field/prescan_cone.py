import torch
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class PrescanCone(Algorithm):
    prescan_n: int = 64

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model, device = utils.init_model(model_name)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

            origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device)
            intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
            intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)
        
        return utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)

    def hit_rate(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        model, device = utils.init_model(model_name)

        hit_rates = []

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            broad_intersections = PrescanCone._broad_scan(model, origins, dirs)

            init_sphere_n = origins.shape[0]
            init_rays_n = init_sphere_n * (init_sphere_n - 1)

            for N in N_values:
                print(N, end='\t')

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device)
                intersections, _ = PrescanCone._targeted_scan(model, origins, dirs)

                sphere_n = origins.shape[0]
                rays_n = sphere_n * (sphere_n - 1)

                hit_rate = intersections.shape[0] / (rays_n + init_rays_n)
                hit_rates.append(hit_rate)
                print(f'{hit_rate:.3f}')

                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()
        
        return hit_rates

    def chamfer(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

            R_values.fill(dirs.shape[0] * dirs.shape[2])

            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, '256')
                intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)
                distance = utils.chamfer_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)

                distances[i] = distance
                print(f'{distance:.6f}')

                del origins, dirs, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances, R_values
    
    def hausdorff(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

            R_values.fill(dirs.shape[0] * dirs.shape[2])

            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, '256')
                intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)
                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)

                distances[i] = distance
                print(f'{distance:.6f}')

                del origins, dirs, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances, R_values

    def time(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)

        times = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values))

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                for j in range(PrescanCone.time_samples):
                    torch.cuda.synchronize()
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
                    broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]

                    origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, '256')
                    intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                    intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)

                    torch.cuda.synchronize()
                    time = timer() - start_time

                    if j == 0:
                        R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])

                    times[i] = times[i] + time
                    print(f'{time:.5f}', end='\t')

                    del origins, dirs, broad_intersections, broad_normals, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                
                print()
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        times = times / PrescanCone.time_samples

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

                for _ in range(PrescanCone.time_samples):
                    torch.cuda.synchronize()
                    broad_start = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
                    intersections = PrescanCone._broad_scan(model, origins, dirs)

                    torch.cuda.synchronize()
                    broad_end = timer()

                    origins, dirs = PrescanCone._generate_cone_rays(intersections, N, device)
                    intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)

                    torch.cuda.synchronize()
                    targeted_end = timer()

                    _ = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)

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

        return times / PrescanCone.time_samples
    
    def time_chamfer(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[float], list[int]]:
        model, device = utils.init_model(model_name)

        times = np.zeros(len(N_values))
        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]

                torch.cuda.synchronize()
                start_time = timer()

                origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
                broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, '256')
                intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)

                torch.cuda.synchronize()
                time = timer() - start_time

                distance = utils.chamfer_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)
                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])

                times[i] = time
                distances[i] = distance
                
                print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')

                del origins, dirs, broad_intersections, broad_normals, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
                
        del model
        torch.cuda.empty_cache()

        return times, distances, R_values
    
    def time_hausdorff(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[float], list[int]]:
        model, device = utils.init_model(model_name)

        times = np.zeros(len(N_values))
        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]

                torch.cuda.synchronize()
                start_time = timer()

                origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
                broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, '256')
                intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)

                torch.cuda.synchronize()
                time = timer() - start_time

                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)
                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])

                times[i] = time
                distances[i] = distance
                
                print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')

                del origins, dirs, broad_intersections, broad_normals, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
                
        del model
        torch.cuda.empty_cache()

        return times, distances, R_values
    
    def optimize(model_name: CheckpointName, N_values: list[int], M_values: list[int]) -> list[list[float]]:   
        model, device = utils.init_model(model_name)

        distances = []

        with torch.no_grad():
            for M in M_values:
                origins, dirs = utils.generate_sphere_rays_tensor(M, device)
                broad_intersections = PrescanCone._broad_scan(model, origins, dirs)
                prescan_distances = []

                for N in N_values:
                    print(f'M: {M}\tN: {N}', end='\t')

                    origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device)
                    intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
            
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)

                    prescan_distances.append(distance)
                    print(f'{distance:.6f}')
                    torch.cuda.empty_cache()
                
                distances.append(prescan_distances)
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances
    
    @staticmethod
    def radius(model_name: CheckpointName, N: int) -> list[float]:
        """Finds the maximum angle between the origins pointing towards (0, 0, 0) and the observed model after a broad scan.

        Args:
            model_name (CheckpointName): Valid model name
            N (int): Number of points to generate along the unit sphere

        Returns:
            list[float]: The maximum angles for all origins (n)
        """
        model, device = utils.init_model(model_name)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            intersections = PrescanCone._broad_scan(model, origins, dirs)

            origins = utils.generate_equidistant_sphere_points_tensor(N, device)
            max_angles = utils.get_max_cone_angles(origins, intersections)

            max_angles = max_angles.cpu().detach().numpy()

            print(f'{np.min(max_angles):.3f}\t{np.max(max_angles):.3f}')

            torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()
        
        return max_angles

    @staticmethod
    def _broad_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """Perform a ray query on the MARF model to get intersections.

        Args:
            model (IntersectionFieldAutoDecoderModel): Initialized MARF model
            origins (torch.Tensor): Origins of the rays (n, 3)
            dirs (torch.Tensor): Ray directions (n, 1, n-1, 3)

        Returns:
            torch.Tensor: Intersection points (m, 3)
        """        
        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

        intersections = result[2]
        intersection_normals = result[3]
        is_intersecting = result[4]

        is_intersecting = is_intersecting.flatten()
        intersections = intersections.flatten(end_dim=2)[is_intersecting]
        intersection_normals = intersection_normals.flatten(end_dim=2)[is_intersecting]

        return intersections, intersection_normals

    @staticmethod
    def _generate_cone_rays(intersections: torch.Tensor, N: int, device: str, m_type: str = 'linear') -> tuple[torch.Tensor, torch.Tensor]:
        """Generates n origins along the unit sphere, finds the maximum angle for each origin
        to cover the intersection points, and generates m random rays within that angle

        Args:
            intersections (torch.Tensor): The intersection points from the broad scan
            N (int): Number of origins, the resulting n is <= N
            device (str): The device to store tensors in
            m_type (str): Which sceme to use when calculating m

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                - Ray origins (n, 3)
                - Ray directions (n, 1, m, 3)
        """ 
        origins = utils.generate_equidistant_sphere_points_tensor(N, device)
        max_angles = utils.get_max_cone_angles(origins, intersections)
        max_angles = max_angles.unsqueeze(1)

        N = origins.shape[0]

        if m_type == 'linear':
            M = N - 1
        else:
            M = int(m_type)

        # Compute basis for origin -> (0, 0, 0)
        central_dirs = -origins
        up = torch.full((N, 3), fill_value=0.0, device=device, dtype=torch.float32)
        up[:, 2] = 1.0
        mask = torch.abs(central_dirs[:, 2]) >= 0.9
        up[mask] = up.new_tensor([1.0, 0.0, 0.0])

        right = torch.cross(central_dirs, up)
        right /= torch.linalg.norm(right, dim=1, keepdim=True)
        up = torch.cross(right, central_dirs)

        # Generate random spherical coordinates within max_angles
        theta = torch.rand(N, M, device=device, dtype=torch.float32) * (2 * torch.pi)
        cos_beta = torch.rand(N, M, device=device, dtype=torch.float32) * (1 - torch.cos(max_angles)) + torch.cos(max_angles)
        sin_beta = torch.sqrt(1 - cos_beta**2).unsqueeze(-1)

        directions = (
            cos_beta.unsqueeze(-1) * central_dirs.unsqueeze(1) +
            sin_beta * torch.cos(theta).unsqueeze(-1) * right.unsqueeze(1) +
            sin_beta * torch.sin(theta).unsqueeze(-1) * up.unsqueeze(1)
        ).unsqueeze(1)

        return origins, directions
    
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
        intersections = intersections.flatten(end_dim=2)[is_intersecting]
        intersection_normals = intersection_normals.flatten(end_dim=2)[is_intersecting]

        return intersections, intersection_normals
    
    @staticmethod
    def _cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals) -> tuple[ndarray, ndarray]:
        intersections = torch.cat((intersections, broad_intersections)).cpu().detach().numpy()
        intersection_normals = torch.cat((intersection_normals, broad_normals)).cpu().detach().numpy()

        return intersections, intersection_normals
