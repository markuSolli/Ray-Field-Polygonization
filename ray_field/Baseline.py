import torch

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import Algorithm, CheckpointName, utils
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
                distance = utils.chamfer_distance_to_stanford(model_name, mesh, Baseline.chamfer_samples)

                distances.append(distance)
                print(f'{distance:.6f}')

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
                distance = utils.hausdorff_distance_to_stanford(model_name, mesh)

                distances.append(distance)
                print(f'{distance:.6f}')
                torch.cuda.empty_cache()

        return distances

    def time(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        model, device = utils.init_model(model_name)

        times = []

        with torch.no_grad():
            for N in N_values:
                print(N, end='\t')

                start_time = timer()

                origins, dirs = utils.generate_sphere_rays(N, device)
                intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)
                _ = utils.poisson_surface_reconstruction(intersections, intersection_normals, Baseline.poisson_depth)
                time = timer() - start_time

                times.append(time)
                print(f'{time:.6f}')
                torch.cuda.empty_cache()

        return times

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

        times = []

        with torch.no_grad():
            for N in N_values:
                print(N, end='\t')

                ray_start = timer()
                origins, dirs = utils.generate_sphere_rays(N, device)
                ray_end = timer()

                intersections, intersection_normals = Baseline._baseline_scan(model, origins, dirs)
                scan_end = timer()

                _ = utils.poisson_surface_reconstruction(intersections, intersection_normals, Baseline.poisson_depth)
                reconstruct_end = timer()

                ray_time = ray_end - ray_start
                scan_time = scan_end - ray_end
                reconstruct_time = reconstruct_end - scan_end

                times.append([ray_time, scan_time, reconstruct_time])
                print(f'{ray_time:.4f}\t{scan_time:.4f}\t{reconstruct_time:.4f}')
                torch.cuda.empty_cache()

        return times
    
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
