import gc
import torch
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class BaselineCPU(Algorithm):

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model = utils.init_model_cpu(model_name)
        origins, dirs = utils.generate_sphere_rays(N, 'cpu')

        with torch.no_grad():
            intersections, intersection_normals = BaselineCPU._baseline_scan(model, origins, dirs)

        return utils.poisson_surface_reconstruction(intersections, intersection_normals, BaselineCPU.poisson_depth)

    def time(model_name: CheckpointName, length: int) -> tuple[list[float], list[int]]:
        model = utils.init_model_cpu(model_name)
        N_values = np.linspace(50, 500, length, dtype=int)

        times = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                for j in range(BaselineCPU.time_samples):
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays(N, 'cpu')
                    intersections, intersection_normals = BaselineCPU._baseline_scan(model, origins, dirs)
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, BaselineCPU.poisson_depth)

                    time = timer() - start_time

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]

                    times[i] = times[i] + time
                    
                    if ((j + 1) % 6) == 0:
                        print(f'{time:.5f}', end='\t')

                    del origins, dirs, intersections, intersection_normals, mesh
                    gc.collect()
                
                print()
                gc.collect()

        del model
        gc.collect()

        times = times / BaselineCPU.time_samples

        return times, R_values

    @staticmethod
    def _baseline_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[ndarray, ndarray]:
        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

        intersections = result[2]
        intersection_normals = result[3]
        is_intersecting = result[4]

        is_intersecting = torch.flatten(is_intersecting)
        intersections = torch.flatten(intersections, end_dim=2)[is_intersecting].detach().numpy()
        intersection_normals = torch.flatten(intersection_normals, end_dim=2)[is_intersecting].detach().numpy()

        return intersections, intersection_normals
