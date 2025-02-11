import trimesh
from ray_field import utils

old_points = utils.generate_equidistant_sphere_points(40)
cuda_points = utils.generate_equidistant_sphere_points_tensor(40, 'cuda')

old_origins, old_dirs = utils.generate_rays_between_points(old_points)
cuda_dirs = utils.generate_rays_between_points_tensor(cuda_points, 'cuda')
old_dirs = old_dirs.detach().numpy()
cuda_points = cuda_points.cpu().detach().numpy()
cuda_dirs = cuda_dirs.cpu().detach().numpy()

print(old_dirs.shape, cuda_dirs.shape)

old_cloud = trimesh.points.PointCloud(old_origins, colors=[0, 0, 255])
cuda_cloud = trimesh.points.PointCloud(cuda_points, colors=[255, 0, 0])

j = 14
paths = []
for i in range(old_dirs.shape[2]):
    paths.append(trimesh.load_path([old_origins[j], old_origins[j] + old_dirs[j,0,i] / 2.0]))
scene = trimesh.Scene([old_cloud, paths])
scene.show()

j = 14
paths = []
for i in range(cuda_dirs.shape[2]):
    paths.append(trimesh.load_path([cuda_points[j], cuda_points[j] + cuda_dirs[j,0,i] / 2.0]))
scene = trimesh.Scene([cuda_cloud, paths])
scene.show()
