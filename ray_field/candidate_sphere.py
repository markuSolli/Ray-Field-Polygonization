import gc
import torch
import numpy as np

from ifield.models.intersection_fields import IntersectionFieldAutoDecoderModel
from ray_field import CheckpointName, utils
from ray_field.algorithm import Algorithm
from open3d.geometry import TriangleMesh
from timeit import default_timer as timer
from numpy import ndarray

class CandidateSphere(Algorithm):
    prescan_n: int = 64
    targeted_m: str = '8'
    chamfer_samples: int = 14

    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        model, device = utils.init_model(model_name)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
            all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)
            radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)

            origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
            intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
        
        return utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)

    def hit_rate(model_name: CheckpointName, length: int) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)

        hit_rates = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
            all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)
            radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)

            R_values.fill(dirs.shape[0] * dirs.shape[2])

            del origins, dirs, all_intersections, all_is_intersecting
            torch.cuda.empty_cache()
            gc.collect()

            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
                intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)

                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
                hit_rate = intersections.shape[0] / R_values[i]

                hit_rates[i] = hit_rate
                print(f'{hit_rate:.5f}')

                del origins, dirs, intersections, intersection_normals
                torch.cuda.empty_cache()
                gc.collect()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return hit_rates, R_values

    def chamfer(model_name: CheckpointName, length: int) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)
        N_end = (250000 - (CandidateSphere.prescan_n ** 2)) / (int(CandidateSphere.targeted_m) * 16)
        N_values = np.linspace(50, N_end, length, dtype=int)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
            all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)
            radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)

            R_values.fill(dirs.shape[0] * dirs.shape[2])

            del origins, dirs, all_intersections, all_is_intersecting
            torch.cuda.empty_cache()
            gc.collect()

            for i in range(len(N_values)):
                N = N_values[i]
                print(f'N: {N}')

                for k in range(CandidateSphere.chamfer_samples):
                    origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
                    intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)

                    if k == 0:
                        R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
            
                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, CandidateSphere.dist_samples)

                    distances[i] = distances[i] + distance

                    del origins, dirs, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()

                torch.cuda.empty_cache()
                gc.collect()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

        distances = distances / CandidateSphere.chamfer_samples

        return distances, R_values

    def hausdorff(model_name: CheckpointName, length: int) -> tuple[list[float], list[int]]:
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)

        distances = np.zeros(len(N_values))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
            all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)
            radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)

            R_values.fill(dirs.shape[0] * dirs.shape[2])

            del origins, dirs, all_intersections, all_is_intersecting
            torch.cuda.empty_cache()
            gc.collect()

            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
                intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)

                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])
        
                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, CandidateSphere.dist_samples)

                distances[i] = distance
                print(f'{distance:.6f}')

                del origins, dirs, intersections, intersection_normals, mesh
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

                for j in range(CandidateSphere.time_samples):
                    torch.cuda.synchronize(device)
                    start_time = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
                    all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)
                    radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]

                    origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
                    intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)

                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)

                    torch.cuda.synchronize(device)
                    time = timer() - start_time

                    if j == 0:
                        R_values[i] = R_values[i] + dirs.shape[0] * dirs.shape[2]

                    times[i] = times[i] + time
                    
                    if ((j + 1) % 6) == 0:
                        print(f'{time:.5f}', end='\t')

                    del origins, dirs, all_intersections, all_is_intersecting, radii, centers, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print()
                torch.cuda.empty_cache()
                gc.collect()

        del model
        torch.cuda.empty_cache()
        gc.collect()

        times = times / CandidateSphere.time_samples

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

                for j in range(CandidateSphere.time_samples):
                    torch.cuda.synchronize(device)
                    coarse_start = timer()

                    origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
                    all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)

                    torch.cuda.synchronize(device)
                    coarse_end = timer()

                    if j == 0:
                        R_values[i] = dirs.shape[0] * dirs.shape[2]
                    
                    radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)
                    origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)

                    torch.cuda.synchronize(device)
                    ray_end = timer()

                    intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)

                    torch.cuda.synchronize(device)
                    targeted_end = timer()

                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)

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

                    del origins, dirs, all_intersections, all_is_intersecting, radii, centers, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print()
                torch.cuda.empty_cache()
                gc.collect()

        del model
        torch.cuda.empty_cache()
        gc.collect()

        times = times / CandidateSphere.time_samples

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

                origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
                broad_intersections, broad_normals, sphere_centers = CandidateSphere._broad_scan(model, origins, dirs)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                radii, centers = CandidateSphere._generate_candidate_spheres(sphere_centers, device)
                origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
                intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = CandidateSphere._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)

                torch.cuda.synchronize()
                time = timer() - start_time

                distance = utils.chamfer_distance_to_stanford(model_name, mesh, CandidateSphere.dist_samples)
                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])

                times[i] = time
                distances[i] = distance
                
                print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')

                del origins, dirs, broad_intersections, broad_normals, sphere_centers, radii, centers, intersections, intersection_normals, mesh
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

                origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
                broad_intersections, broad_normals, sphere_centers = CandidateSphere._broad_scan(model, origins, dirs)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                radii, centers = CandidateSphere._generate_candidate_spheres(sphere_centers, device)
                origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
                intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
                intersections, intersection_normals = CandidateSphere._cat_and_move(intersections, intersection_normals, broad_intersections, broad_normals)

                mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)

                torch.cuda.synchronize()
                time = timer() - start_time

                distance = utils.hausdorff_distance_to_stanford(model_name, mesh, CandidateSphere.dist_samples)
                R_values[i] = R_values[i] + (dirs.shape[0] * dirs.shape[2])

                times[i] = time
                distances[i] = distance
                
                print(f'N: {N}\tTime: {time:.5f}\tDistance: {distance:.5f}')

                del origins, dirs, broad_intersections, broad_normals, sphere_centers, radii, centers, intersections, intersection_normals, mesh
                torch.cuda.empty_cache()
                        
        del model
        torch.cuda.empty_cache()

        return times, distances, R_values
    
    def optimize(model_name: CheckpointName, length: int, M_values: list[int]) -> tuple[list[list[float]], list[list[int]]]:   
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)

        distances = np.zeros((len(M_values), length))
        R_values = np.zeros((len(M_values), length), dtype=int)

        with torch.no_grad():
            for i in range(len(M_values)):
                M = M_values[i]
                origins, dirs = utils.generate_sphere_rays_tensor(M, device)
                all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)
                radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)
                rays_n = dirs.shape[0] * dirs.shape[2]
                R_values[i].fill(rays_n)
                N_end = int((250000 - rays_n) / (int(CandidateSphere.targeted_m) * 16))

                N_values = np.linspace(50, N_end, length, dtype=int)

                del all_intersections, all_is_intersecting
                torch.cuda.empty_cache()
                gc.collect()

                for j in range(len(N_values)):
                    N = N_values[j]
                    print(f'M: {M}\tN: {N}')

                    for k in range(CandidateSphere.chamfer_samples):
                        origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
                        intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
                
                        mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
                        distance = utils.chamfer_distance_to_stanford(model_name, mesh, CandidateSphere.dist_samples)

                        distances[i][j] = distances[i][j] + distance

                        if k == 0:
                            R_values[i][j] = R_values[i][j] + dirs.shape[0] * dirs.shape[2]

                        del origins, dirs, intersections, intersection_normals, mesh
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                
                del radii, centers
                torch.cuda.empty_cache()
                gc.collect()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

        distances = distances / CandidateSphere.chamfer_samples

        return distances, R_values
    
    def optimize_ray(model_name: CheckpointName, length: int, M_values: list[str]) -> tuple[list[list[float]], list[list[int]]]:   
        model, device = utils.init_model(model_name)

        distances = np.zeros((len(M_values), length))
        R_values = np.zeros((len(M_values), length), dtype=int)

        origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
        all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)
        radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)
        rays_n = dirs.shape[0] * dirs.shape[2]
        R_values.fill(rays_n)

        del all_intersections, all_is_intersecting
        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            for i in range(len(M_values)):
                M = M_values[i]

                if M == 'n/16':
                    N_end = 500
                else:
                    N_end = int((250000 - rays_n) / (int(M) * 16))
                
                N_values = np.linspace(50, N_end, length, dtype=int)

                for j in range(length):
                    N = N_values[j]
                    print(f'M: {M}\tN: {N}')

                    for k in range(CandidateSphere.chamfer_samples):
                        origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, M)
                        intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)
                
                        mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
                        distance = utils.chamfer_distance_to_stanford(model_name, mesh, CandidateSphere.dist_samples)

                        distances[i][j] = distances[i][j] + distance

                        if k == 0:
                            R_values[i][j] = R_values[i][j] + dirs.shape[0] * dirs.shape[2]

                        del origins, dirs, intersections, intersection_normals, mesh
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                
                torch.cuda.empty_cache()
                gc.collect()
        
        del radii, centers, model
        torch.cuda.empty_cache()
        gc.collect()

        distances = distances / CandidateSphere.chamfer_samples

        return distances, R_values
    
    @staticmethod
    def dist_deviation(model_name: CheckpointName, length: int, samples: int) -> tuple[list[list[float]], list[int]]:
        model, device = utils.init_model(model_name)
        N_values = np.linspace(50, 1900, length, dtype=int)

        distances = np.zeros((len(N_values), samples))
        R_values = np.zeros(len(N_values), dtype=int)

        with torch.no_grad():
            for i in range(len(N_values)):
                N = N_values[i]
                print(N, end='\t')

                origins, dirs = utils.generate_sphere_rays_tensor(CandidateSphere.prescan_n, device)
                all_intersections, all_is_intersecting = CandidateSphere._broad_scan(model, origins, dirs)
                radii, centers = CandidateSphere._generate_candidate_spheres(all_intersections, all_is_intersecting)

                R_values[i] = dirs.shape[0] * dirs.shape[2]

                for j in range(samples):
                    origins, dirs = CandidateSphere._generate_candidate_rays(radii, centers, N, device, CandidateSphere.targeted_m)
                    intersections, intersection_normals = CandidateSphere._targeted_scan(model, origins, dirs)

                    mesh = utils.poisson_surface_reconstruction(intersections, intersection_normals, CandidateSphere.poisson_depth)
                    distance = utils.chamfer_distance_to_stanford(model_name, mesh, CandidateSphere.dist_samples)

                    distances[i][j] = distance
                    if j == 0:
                        R_values[i] = R_values[i] + dirs.shape[0] * dirs.shape[2]

                    if ((j + 1) % 6) == 0:
                        print(f'{distance:.5f}', end='\t')

                    del origins, dirs, intersections, intersection_normals, mesh
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print()
                del all_intersections, all_is_intersecting, radii, centers
                torch.cuda.empty_cache()
                gc.collect()

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return distances, R_values

    @staticmethod
    def _broad_scan(model: IntersectionFieldAutoDecoderModel, origins: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:           
        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False, return_all_atoms = True)

        all_intersections = result[8]
        all_is_intersecting = result[12]

        all_intersections = all_intersections.flatten(end_dim=2)
        all_is_intersecting = all_is_intersecting.flatten(end_dim=2)

        return all_intersections, all_is_intersecting

    @staticmethod
    def _generate_candidate_spheres(intersections: torch.Tensor, is_intersecting: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_intersecting = is_intersecting.unsqueeze(-1)

        # Find candidate centers
        intersection_sums = torch.sum(intersections * is_intersecting, dim=0)
        valid_count = torch.sum(is_intersecting, dim=0).clamp(min=1)
        candidate_centers = intersection_sums / valid_count

        # Find candidate radii
        is_intersecting = is_intersecting.squeeze()
        distances = torch.norm(intersections - candidate_centers, dim=2)
        distances[~is_intersecting] = 0
        max_radii = distances.max(dim=0).values

        return max_radii, candidate_centers

    @staticmethod
    def _generate_candidate_rays(radii: torch.Tensor, centers: torch.Tensor, N: int, device: str, m_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates n origins along the unit sphere. For each candidate sphere,
        M rays are generated in a cone that covers that sphere. The value of M is calculated through the chosen approach.

        Args:
            radii (torch.Tensor): Sphere radii (16)
            centers (torch.Tensor): Sphere centers (16, 3)
            N (int): Number of origins, the resulting n is <= N
            device (str): The device to store tensors in
            m_type (str): The approach for calculating M.

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                - Ray origins (n, 3)
                - Ray directions (n, 1, m, 3)
        """        
        origins = utils.generate_equidistant_sphere_points_tensor(N, device)
        N = origins.shape[0]

        if m_type == 'n/16':
            M = N // 16
        else:
            M = int(m_type)

        # Compute basis for origin -> sphere center
        central_dirs = centers[None, :, :] - origins[:, None, :]
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).expand(N, 16, 3).clone()
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
        theta = torch.rand(N, 16, M, device=device) * (2 * torch.pi)
        beta = torch.rand(N, 16, M, device=device) * max_angles
        cos_beta = torch.cos(beta).unsqueeze(-1)
        sin_beta = torch.sin(beta).unsqueeze(-1)

        directions = (
            cos_beta * central_dirs.unsqueeze(2) +
            sin_beta * torch.cos(theta).unsqueeze(-1) * right.unsqueeze(2) +
            sin_beta * torch.sin(theta).unsqueeze(-1) * up.unsqueeze(2)
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
