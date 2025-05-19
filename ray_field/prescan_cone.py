import gc
import torch
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class PrescanCone(Algorithm):
    prescan_n: int = 32
    targeted_m: str = '128'

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model, device = utils.init_model(model_name)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

            origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)
            intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
            intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)
        
        return utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)

    def hit_rate(model_name: CheckpointName, length: int) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)

        hit_rates = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

            R_values.fill(dirs.shape[0] * dirs.shape[2])

            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)
                intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
                hit_rate = intersections.shape[0] / R_values[i]

                hit_rates[i] = hit_rate
                print(f'{hit_rate:.5f}')

                del origins, dirs, intersections, intersection_normals
                torch.cuda.empty_cache()
                gc.collect()
            
            del broad_intersections, broad_normals
            torch.cuda.empty_cache()
            gc.collect()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return hit_rates, R_values

    def chamfer(model_name: CheckpointName, length: int) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

            R_values.fill(dirs.shape[0] * dirs.shape[2])

            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)
                intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)
                distance = utils.chamfer_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)

                distances[i] = distance
                print(f'{distance:.5f}')

                del origins, dirs, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
                gc.collect()
            
            del broad_intersections, broad_normals
            torch.cuda.empty_cache()
            gc.collect()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return distances, R_values
    
    def hausdorff(model_name: CheckpointName, length: int) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
            broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

            R_values.fill(dirs.shape[0] * dirs.shape[2])

            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)
                intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)
                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)

                distances[i] = distance
                print(f'{distance:.5f}')

                del origins, dirs, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
                gc.collect()
            
            del broad_intersections, broad_normals
            torch.cuda.empty_cache()
            gc.collect()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return distances, R_values

    def time(model_name: CheckpointName, length: int) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)

        times = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                for j in range(PrescanCone.time_samples):
                    torch.cuda.synchronize(device)
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
                    broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]

                    origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)
                    intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                    intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)

                    torch.cuda.synchronize(device)
                    time = timer() - start_time

                    if j == 0:
                        R_values[i] = R_values[i] + dirs.shape[0] * dirs.shape[2]

                    times[i] = times[i] + time
                    
                    if ((j + 1) % 6) == 0:
                        print(f'{time:.5f}', end='\t')

                    del origins, dirs, broad_intersections, broad_normals, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print()
                torch.cuda.empty_cache()
                gc.collect()

        del model
        torch.cuda.empty_cache()
        gc.collect()

        times = times / PrescanCone.time_samples

        return times, R_values

    def time_steps(model_name: CheckpointName, length: int) -> tuple[list[str], list[list[float]], list[int]]:
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)
        steps = ['Coarse Scan', 'Ray generation', 'MARF Query', 'PSR']

        times = np.zeros((len(steps), len(N_values)))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                for j in range(PrescanCone.time_samples):
                    torch.cuda.synchronize(device)
                    coarse_start = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
                    broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

                    torch.cuda.synchronize(device)
                    coarse_end = timer()

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]

                    origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)

                    torch.cuda.synchronize(device)
                    ray_end = timer()

                    intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                    intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                    torch.cuda.synchronize(device)
                    targeted_end = timer()

                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)

                    torch.cuda.synchronize(device)
                    psr_end = timer()

                    if j == 0:
                        R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])

                    times[0][i] = times[0][i] + (coarse_end - coarse_start)
                    times[1][i] = times[1][i] + (ray_end - coarse_end)
                    times[2][i] = times[2][i] + (targeted_end - ray_end)
                    times[3][i] = times[3][i] + (psr_end - targeted_end)
                    
                    if ((j + 1) % 5) == 0:
                        print(f'{(psr_end - coarse_start):.5f}', end='\t')

                    del origins, dirs, broad_intersections, broad_normals, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print()
                torch.cuda.empty_cache()
                gc.collect()

        del model
        torch.cuda.empty_cache()
        gc.collect()

        times = times / PrescanCone.time_samples

        return steps, times, R_values
    
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

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)
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

                origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)
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
    
    def optimize(model_name: CheckpointName, N_values: list[int], M_values: list[int]) -> tuple[list[list[float]], list[list[int]]]:   
        model, device = utils.init_model(model_name)

        distances = np.zeros((len(M_values), len(N_values)))
        R_values = np.zeros((len(M_values), len(N_values)), dtype=int)

        with torch.no_grad():
            for i in range(len(M_values)):
                M = M_values[i]
                origins, dirs = utils.generate_sphere_rays_tensor(M, device)
                broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

                R_values[i].fill(dirs.shape[0] * dirs.shape[2])

                for j in range(len(N_values)):
                    N = N_values[j]
                    print(f'M: {M}\tN: {N}', end='\t')

                    origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, PrescanCone.targeted_m)
                    intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                    intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)
            
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)

                    distances[i][j] = distance
                    R_values[i][j] = R_values[i][j] + dirs.shape[0] * dirs.shape[2]
                    
                    print(f'{distance:.6f}')

                    del origins, dirs, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()
                
                del broad_intersections, broad_normals
                torch.cuda.empty_cache()
                gc.collect()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return distances, R_values
    
    def optimize_ray(model_name: CheckpointName, N_values: list[list[int]], M_values: list[str]) -> tuple[list[list[float]], list[list[int]]]:   
        model, device = utils.init_model(model_name)

        distances = np.zeros((len(M_values), len(N_values[0])))
        R_values = np.zeros((len(M_values), len(N_values[0])), dtype=int)

        origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
        broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

        R_values.fill(dirs.shape[0] * dirs.shape[2])

        with torch.no_grad():
            for i in range(len(M_values)):
                M = M_values[i]

                for j in range(len(N_values[i])):
                    N = N_values[i][j]
                    print(f'M: {M}\tN: {N}', end='\t')

                    origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, M)
                    intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                    intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)
            
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)

                    distances[i][j] = distance
                    R_values[i][j] = R_values[i][j] + dirs.shape[0] * dirs.shape[2]
                    
                    print(f'{distance:.6f}')

                    del origins, dirs, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()
                
                torch.cuda.empty_cache()
                gc.collect()
        
        del broad_intersections, broad_normals, model
        torch.cuda.empty_cache()
        gc.collect()

        return distances, R_values
    
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
    def dist_deviation(model_name: CheckpointName, N_values: list[int], samples: int) -> tuple[list[list[float]], list[int]]:
        model, device = utils.init_model(model_name)

        distances = np.zeros((len(N_values), samples))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = utils.generate_sphere_rays_tensor(PrescanCone.prescan_n, device)
                broad_intersections, broad_normals = PrescanCone._broad_scan(model, origins, dirs)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                for j in range(samples):
                    origins, dirs = PrescanCone._generate_cone_rays(broad_intersections, N, device, '256')
                    intersections, intersection_normals = PrescanCone._targeted_scan(model, origins, dirs)
                    intersections, intersection_normals = PrescanCone._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)
            
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, PrescanCone.poisson_depth)
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, PrescanCone.dist_samples)

                    distances[i][j] = distance
                    if j == 0:
                        R_values[i] = R_values[i] + dirs.shape[0] * dirs.shape[2]

                    if ((j + 1) % 6) == 0:
                        print(f'{distance:.5f}', end='\t')

                    del origins, dirs, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print()
                del broad_intersections, broad_normals
                torch.cuda.empty_cache()
                gc.collect()

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return distances, R_values

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
    def _generate_cone_rays(intersections: torch.Tensor, N: int, device: str, m_type: str) -> tuple[torch.Tensor, torch.Tensor]:
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

        if m_type == 'n':
            M = N
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
        beta = torch.rand(N, M, device=device, dtype=torch.float32) * max_angles
        cos_beta = torch.cos(beta).unsqueeze(-1)
        sin_beta = torch.sin(beta).unsqueeze(-1)

        directions = (
            cos_beta * central_dirs.unsqueeze(1) +
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
