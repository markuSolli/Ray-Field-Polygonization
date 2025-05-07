import torch
import trimesh

from ray_field import utils

with torch.no_grad():
    model, device = utils.init_model('bunny')

    origins, dirs = utils.generate_sphere_rays_tensor(50, device)

    result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False, return_all_atoms = True)

    all_intersections = result[8]
    all_normals = result[9]
    all_is_intersecting = result[12]

    all_is_intersecting = all_is_intersecting.flatten(end_dim=2)
    all_intersections = all_intersections.flatten(end_dim=2)

    print(all_intersections.shape, all_is_intersecting.shape)
    exit()

    intersect_cloud = trimesh.points.PointCloud(all_intersections)
    scene = trimesh.Scene([intersect_cloud])
    scene.show()
