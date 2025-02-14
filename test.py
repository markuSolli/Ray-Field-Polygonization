import torch
from ray_field import utils

model, device = utils.init_model('bunny')
with torch.no_grad():
    origins, dirs = utils.generate_sphere_rays_tensor(60, device)

    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False, return_all_atoms = True)

    depths = result[0].cpu()
    silhouettes = result[1].cpu()
    intersections = result[2].cpu()
    intersection_normals = result[3].cpu()
    is_intersecting = result[4].cpu()
    sphere_centers = result[5].cpu()
    sphere_radii = result[6].cpu()
    
    atom_indices = result[7].cpu()
    all_intersections = result[8].cpu()
    all_intersection_normals = result[9].cpu()
    all_depths = result[10].cpu()
    all_silhouettes = result[11].cpu()
    all_is_intersecting = result[12].cpu()
    all_sphere_centers = result[13].cpu()
    all_sphere_radii = result[14].cpu()

    print(f'depths:\t\t\t\t{depths.shape}')
    print(f'silhouettes:\t\t\t{silhouettes.shape}')
    print(f'intersections:\t\t\t{intersections.shape}')
    print(f'intersection_normals:\t\t{intersection_normals.shape}')
    print(f'is_intersecting:\t\t{is_intersecting.shape}')
    print(f'sphere_centers:\t\t\t{sphere_centers.shape}')

    print(f'sphere_radii:\t\t\t{sphere_radii.shape}')
    print(f'atom_indices:\t\t\t{atom_indices.shape}')
    print(f'all_intersections:\t\t{all_intersections.shape}')
    print(f'all_intersection_normals:\t{all_intersection_normals.shape}')
    print(f'all_depths:\t\t\t{all_depths.shape}')
    print(f'all_silhouettes:\t\t{all_silhouettes.shape}')
    print(f'all_is_intersecting:\t\t{all_is_intersecting.shape}')
    print(f'all_sphere_centers:\t\t{all_sphere_centers.shape}')
    print(f'all_sphere_radii:\t\t{all_sphere_radii.shape}')
