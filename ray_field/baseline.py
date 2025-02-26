import torch
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class Baseline(Algorithm):

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model, device = utils.init_model(model_name)
        origins, dirs = utils.generate_sphere_rays(N, device)

        with torch.no_grad():
            intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)

        return utils.poisson_surface_reconstruction(intersections, intersection_normals, Baseline.poisson_depth)

    def hit_rate(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        model, device = utils.init_model(model_name)

        hit_rates = []

        with torch.no_grad():
            for N in N_values:
                print(N, end='\t')
                origins, dirs = utils.generate_sphere_rays(N, device)

                sphere_n = origins.shape[0]
                rays_n = sphere_n * (sphere_n - 1)

                intersections, _ = Baseline._baseline_scan(model, origins, dirs)
                
                hit_rate = intersections.shape[0] / rays_n
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
            for N in N_values:
                print(N, end='\t')

                origins, dirs = utils.generate_sphere_rays(N, device)
                intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, Baseline.poisson_depth)
                distance = utils.chamfer_distance_to_stanford(model_name, mesh, Baseline.dist_samples)

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
            for N in N_values:
                print(N, end='\t')

                origins, dirs = utils.generate_sphere_rays(N, device)
                intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, Baseline.poisson_depth)
                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, Baseline.dist_samples)

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

                for j in range(Baseline.time_samples):
                    torch.cuda.synchronize()
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays(N, device)
                    intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)
                    _ = utils.poisson_surface_reconstruction(intersections, intersection_normals, Baseline.poisson_depth)

                    torch.cuda.synchronize()
                    time = timer() - start_time

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]

                    times[i] = times[i] + time
                    print(f'{time:.5f}', end='\t')
                    torch.cuda.empty_cache()
                
                print()
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        times = times / Baseline.time_samples

        return times, R_values

    def time_steps(model_name: CheckpointName, N_values: list[int]) -> list[list[float]]:
        """Measure execution time of the surface reconstruction divided in
            - Ray generation
            - MARF query
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

                for _ in range(Baseline.time_samples):
                    torch.cuda.synchronize()
                    ray_start = timer()

                    origins, dirs = utils.generate_sphere_rays(N, device)

                    torch.cuda.synchronize()
                    ray_end = timer()

                    intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)

                    torch.cuda.synchronize()
                    scan_end = timer()

                    _ = utils.poisson_surface_reconstruction(intersections, intersection_normals, Baseline.poisson_depth)

                    torch.cuda.synchronize()
                    reconstruct_end = timer()

                    ray_time = ray_end - ray_start
                    scan_time = scan_end - ray_end
                    reconstruct_time = reconstruct_end - scan_end

                    times[i][0] = times[i][0] + ray_time
                    times[i][1] = times[i][1] + scan_time
                    times[i][2] = times[i][2] + reconstruct_time
                    print(f'N: {N}\t{ray_time:.4f}\t{scan_time:.4f}\t{reconstruct_time:.4f}')
                    torch.cuda.empty_cache()
                
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

        return times / Baseline.time_samples
    
    def time_chamfer(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[float]]:
        model, device = utils.init_model(model_name)

        times = np.zeros(len(N_values))
        distances = np.zeros(len(N_values))

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]

                for _ in range(Baseline.time_samples):
                    torch.cuda.synchronize()
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays(N, device)
                    intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, Baseline.poisson_depth)

                    torch.cuda.synchronize()
                    time = timer() - start_time
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, Baseline.dist_samples)

                    times[i] = times[i] + time
                    distances[i] = distances[i] + distance
                    print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')
                    torch.cuda.empty_cache()
                
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        times = times / Baseline.time_samples
        distances = distances / Baseline.time_samples

        return times, distances
    
    def optimize(model_name: CheckpointName, N_values: list[int], M_values: list[int]) -> list[list[float]]:
        model, device = utils.init_model(model_name)

        distances = []

        with torch.no_grad():
            for N in N_values:
                origins, dirs = utils.generate_sphere_rays(N, device)
                intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)
                depth_distances = []

                for M in M_values:
                    print(f'N: {N}\tM: {M}', end='\t')
            
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, M)
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, Baseline.dist_samples)

                    depth_distances.append(distance)
                    print(f'{distance:.6f}')
                    torch.cuda.empty_cache()
                
                distances.append(depth_distances)
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        distances = np.array(distances)

        return distances.T
    
    @staticmethod
    def radius(N: int) -> float:
        """Finds the maximum angle between the origins pointing towards (0, 0, 0) and another origin.

        Args:
            N (int): Number of points to generate along the unit sphere

        Returns:
            float: The maximum angle
        """        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        sphere_points = utils.generate_equidistant_sphere_points(N)
        first_point = sphere_points[0]
        other_points = sphere_points[1:]
        
        first_point = torch.from_numpy(first_point).unsqueeze(0).to(device)
        other_points = torch.from_numpy(other_points).to(device)

        max_angles = utils.get_max_cone_angles(first_point, other_points)
        max_angles = max_angles.cpu().detach().numpy()
        
        return max_angles[0]

    @staticmethod
    def _baseline_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[ndarray, ndarray]:
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

        is_intersecting = torch.flatten(is_intersecting)
        intersections = torch.flatten(intersections, end_dim=2)[is_intersecting].cpu().detach().numpy()
        intersection_normals = torch.flatten(intersection_normals, end_dim=2)[is_intersecting].cpu().detach().numpy()

        return intersections, intersection_normals
