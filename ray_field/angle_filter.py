import torch
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class AngleFilter(Algorithm):

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model, device = utils.init_model(model_name)
        origins, dirs = utils.generate_sphere_rays_tensor(N, device)

        with torch.no_grad():
            intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
            intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting)

        return utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)

    def hit_rate(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        pass

    def chamfer(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = utils.generate_sphere_rays_tensor(N, device)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
                intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting)
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)
                distance = utils.chamfer_distance_to_stanford(model_name, mesh, AngleFilter.dist_samples)

                distances[i] = distance
                print(f'{distance:.6f}')

                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances, R_values
    
    def hausdorff(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = utils.generate_sphere_rays_tensor(N, device)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
                intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting)
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)
                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, AngleFilter.dist_samples)

                distances[i] = distance
                print(f'{distance:.6f}')

                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances, R_values

    def time(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)

        times = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                for j in range(AngleFilter.time_samples):
                    torch.cuda.synchronize(device)
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(N, device)

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]
                    
                    intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
                    intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting)
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)

                    torch.cuda.synchronize(device)
                    time = timer() - start_time

                    times[i] = times[i] + time
                    print(f'{time:.5f}', end='\t')

                    del origins, dirs, intersections, intersection_normals, is_intersecting, mesh
                    torch.cuda.empty_cache()
                
                print()
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

        times = times / AngleFilter.time_samples

        return times, R_values

    def time_steps(model_name: CheckpointName, N_values: list[int]) -> list[list[float]]:
        pass
    
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

                origins, dirs = utils.generate_sphere_rays_tensor(N, device)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
                intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting)
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)

                torch.cuda.synchronize()
                time = timer() - start_time

                distance = utils.chamfer_distance_to_stanford(model_name, mesh, AngleFilter.dist_samples)

                times[i] = time
                distances[i] = distance
                
                print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')

                del origins, dirs, intersections, intersection_normals, mesh
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

                origins, dirs = utils.generate_sphere_rays_tensor(N, device)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
                intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting)
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)

                torch.cuda.synchronize()
                time = timer() - start_time

                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, AngleFilter.dist_samples)

                times[i] = time
                distances[i] = distance
                
                print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')

                del origins, dirs, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
                        
        del model
        torch.cuda.empty_cache()

        return times, distances, R_values
    
    def chamfer_marf(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        marf_intersections = utils.chamfer_distance_to_marf_1(model_name)

        print(f'{model_name}\t{marf_intersections.shape[0]}')

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]

                origins, dirs = utils.generate_sphere_rays_tensor(N, device)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
                intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting)
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)
                distance = utils.chamfer_distance_to_marf_2(intersections, mesh)

                distances[i] = distance
                print(f'N: {N}\t{distance:.6f}')

                del origins, dirs, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
        
        del model, marf_intersections
        torch.cuda.empty_cache()

        return distances, R_values
    
    @staticmethod
    def optimize(model_name: CheckpointName, N_values: list[int], M_values: ndarray) -> tuple[ndarray, ndarray]:
        model, device = utils.init_model(model_name)

        distances = np.zeros((len(N_values), len(M_values)))
        R_values = np.zeros((len(N_values), len(M_values)), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]

                for j in range(len(M_values)):
                    M = M_values[j]
                    
                    origins, dirs = utils.generate_sphere_rays_tensor(N, device)

                    R_values[i][j] = dirs.shape[0] * dirs.shape[2]

                    intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
                    intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting, M)

                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, AngleFilter.dist_samples)

                    distances[i][j] = distance
                    print(f'N: {N}\tM: {M:.1f}\t{distance:.6f}')

                    del origins, dirs, intersections, intersection_normals, is_intersecting, mesh
                    torch.cuda.empty_cache()
                
                torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        return distances.T, R_values.T
    
    @staticmethod
    def optimize_time(model_name: CheckpointName, N_values: list[int], M_values: ndarray) -> tuple[ndarray, ndarray]:
        model, device = utils.init_model(model_name)

        times = np.zeros((len(N_values), len(M_values)))
        R_values = np.zeros((len(N_values), len(M_values)), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]

                for j in range(len(M_values)):
                    M = M_values[j]
                    print(f'N: {N}\tM{M:.1f}', end='\t')

                    for k in range(AngleFilter.time_samples):
                        torch.cuda.synchronize(device)
                        start_time = timer()
                        
                        origins, dirs = utils.generate_sphere_rays_tensor(N, device)

                        if (k == 0):
                            R_values[i][j] = dirs.shape[0] * dirs.shape[2]

                        intersections, intersection_normals, is_intersecting = AngleFilter._baseline_scan(model, origins, dirs)
                        intersections, intersection_normals = AngleFilter._filter(dirs, intersections, intersection_normals, is_intersecting, M)

                        mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, AngleFilter.poisson_depth)

                        torch.cuda.synchronize(device)
                        time = timer() - start_time

                        times[i][j] = times[i][j] + time
                        print(f'{time:.6f}', end='\t')

                        del origins, dirs, intersections, intersection_normals, is_intersecting, mesh
                        torch.cuda.empty_cache()
                
                    torch.cuda.empty_cache()
                    print()
                
                torch.cuda.empty_cache()
            
            torch.cuda.empty_cache()
        
        del model
        torch.cuda.empty_cache()

        times = times / AngleFilter.time_samples

        return times.T, R_values.T

    @staticmethod
    def _baseline_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:     
        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

        intersections = result[2].squeeze()
        intersection_normals = result[3].squeeze()
        is_intersecting = result[4].squeeze()

        return intersections, intersection_normals, is_intersecting

    @staticmethod
    def _filter(dirs: torch.Tensor, intersections: torch.Tensor, intersection_normals: torch.Tensor, is_intersecting: torch.Tensor, limit: float = -0.707) -> tuple[ndarray, ndarray]:     
        dirs = dirs.squeeze()
        cosine = torch.sum(dirs * intersection_normals, dim=-1)
        mask = (cosine < limit) & is_intersecting

        intersections = intersections[mask].cpu().detach().numpy()
        intersection_normals = intersection_normals[mask].cpu().detach().numpy()

        return intersections, intersection_normals
