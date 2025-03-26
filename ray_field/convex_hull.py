import torch
import trimesh
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from trimesh import Trimesh
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class ConvexHull(Algorithm):

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model, device = utils.init_model(model_name)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(N, device)
            intersections, atom_indices = ConvexHull._baseline_scan(model, origins, dirs)

        return ConvexHull._construct_convex_hull(intersections, atom_indices)

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
                intersections, atom_indices = ConvexHull._baseline_scan(model, origins, dirs)
                mesh = ConvexHull._construct_convex_hull(intersections, atom_indices)
                distance = utils.chamfer_distance_to_stanford_trimesh(model_name, mesh, ConvexHull.dist_samples)

                R_values[i] = dirs.shape[0] * dirs.shape[2]
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
                intersections, atom_indices = ConvexHull._baseline_scan(model, origins, dirs)
                mesh = ConvexHull._construct_convex_hull(intersections, atom_indices)
                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, ConvexHull.dist_samples)

                R_values[i] = dirs.shape[0] * dirs.shape[2]
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

                for j in range(ConvexHull.time_samples):
                    torch.cuda.synchronize(device)
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(N, device)
                    intersections, atom_indices = ConvexHull._baseline_scan(model, origins, dirs)
                    mesh = ConvexHull._construct_convex_hull(intersections, atom_indices)

                    torch.cuda.synchronize(device)
                    time = timer() - start_time

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]

                    times[i] = times[i] + time
                    print(f'{time:.5f}', end='\t')

                    del origins, dirs, intersections, atom_indices, mesh
                    torch.cuda.empty_cache()
                
                print()
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

        times = times / ConvexHull.time_samples

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
                intersections, atom_indices = ConvexHull._baseline_scan(model, origins, dirs)
                mesh = ConvexHull._construct_convex_hull(intersections, atom_indices)

                torch.cuda.synchronize()
                time = timer() - start_time

                distance = utils.chamfer_distance_to_stanford(model_name, mesh, ConvexHull.dist_samples)
                R_values[i] = dirs.shape[0] * dirs.shape[2]

                times[i] = time
                distances[i] = distance
                
                print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')

                del origins, dirs, intersections, atom_indices, mesh
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
                intersections, atom_indices = ConvexHull._baseline_scan(model, origins, dirs)
                mesh = ConvexHull._construct_convex_hull(intersections, atom_indices)

                torch.cuda.synchronize()
                time = timer() - start_time

                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, ConvexHull.dist_samples)
                R_values[i] = dirs.shape[0] * dirs.shape[2]

                times[i] = time
                distances[i] = distance
                
                print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')

                del origins, dirs, intersections, atom_indices, mesh
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
                intersections, atom_indices = ConvexHull._baseline_scan(model, origins, dirs)
                mesh = ConvexHull._construct_convex_hull(intersections, atom_indices)
                distance = utils.chamfer_distance_to_marf_2(intersections, mesh)

                R_values[i] = dirs.shape[0] * dirs.shape[2]
                distances[i] = distance
                print(f'N: {N}\t{distance:.6f}')

                del origins, dirs, intersections, atom_indices, mesh
                torch.cuda.empty_cache()
        
        del model, marf_intersections
        torch.cuda.empty_cache()

        return distances, R_values

    @staticmethod
    def _baseline_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[ndarray, ndarray]:     
        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False, return_all_atoms = True)
 
        intersections = result[2]
        is_intersecting = result[4]
        atom_indices = result[7]
    
        is_intersecting = is_intersecting.flatten()
        intersections = intersections.flatten(end_dim=2)[is_intersecting].cpu().detach()
        atom_indices = atom_indices.flatten()[is_intersecting].cpu().detach()
    
        return intersections, atom_indices
    
    @staticmethod
    def _construct_convex_hull(intersections: torch.Tensor, atom_indices: torch.Tensor) -> Trimesh:     
        unique_atoms = torch.unique(atom_indices)
        convex_hulls = []

        for atom in unique_atoms:
            mask = (atom_indices == atom)
            points = intersections[mask]
            
            if points.shape[0] > 3:
                cloud = trimesh.points.PointCloud(points)
                convex_hulls.append(cloud.convex_hull)
        
        return trimesh.util.concatenate(convex_hulls)
