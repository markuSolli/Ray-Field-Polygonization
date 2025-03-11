import torch
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class Shrinkwrap(Algorithm):

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model, device = utils.init_model(model_name)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(N, device)
            broad_intersections, broad_normals = Shrinkwrap._broad_scan(model, origins, dirs)

            origins, dirs = Shrinkwrap._generate_targeted_rays(broad_intersections, broad_normals, device)
            intersections, intersection_normals = Shrinkwrap._targeted_scan(model, origins, dirs)
            intersections, intersection_normals = Shrinkwrap._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)
        
        return utils.poisson_surface_reconstruction(intersections, intersection_normals, Shrinkwrap.poisson_depth)

    def hit_rate(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        pass

    def chamfer(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        N_values = list(range(10, 41, 5))
        model, device = utils.init_model(model_name)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = utils.generate_sphere_rays_tensor(N, device)
                broad_intersections, broad_normals = Shrinkwrap._broad_scan(model, origins, dirs)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                origins, dirs = Shrinkwrap._generate_targeted_rays(broad_intersections, broad_normals, device)
                intersections, intersection_normals = Shrinkwrap._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = Shrinkwrap._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, Shrinkwrap.poisson_depth)
                distance = utils.chamfer_distance_to_stanford(model_name, mesh, Shrinkwrap.dist_samples)

                distances[i] = distance
                print(f'{distance:.6f}')

                del origins, dirs, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances, R_values
    
    def hausdorff(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        pass

    def time(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        pass

    def time_steps(model_name: CheckpointName, N_values: list[int]) -> list[list[float]]:
        pass
    
    def time_chamfer(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[float], list[int]]:
        pass
    
    def time_hausdorff(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[float], list[int]]:
        pass
    
    def optimize(model_name: CheckpointName, N_values: list[int], M_values: list[int]) -> list[list[float]]:   
        pass

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
    def _generate_targeted_rays(intersections: torch.Tensor, normals: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        origins = intersections + 0.1 * normals
        dirs = utils.generate_rays_between_points_tensor(origins, device)

        return origins, dirs
    
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
