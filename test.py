import trimesh
import torch
import utils
from ifield.models import intersection_fields

checkpoint = 'MARF/experiments/logdir/tensorboard/experiment-stanfordv12-bunny-both2marf-16atom-50xinscr-10dmiss-geom-25cnrml-8x512fc-leaky_relu-hit-0minatomstdngxp-500sphgrow-10mdrop-layernorm-multi_view-10dmv-nocond-100cwu500clr70tvs-2023-05-31-0010-nqzh/checkpoints/epoch=197-step=13860.ckpt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(checkpoint)
model.eval().to(device)

origins, dirs = utils.generate_rays_between_sphere_points(400)
origins = origins.to(device)
dirs = dirs.to(device)

result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

#depths = result[0].cpu()
#silhouettes = result[1].cpu()
intersections = result[2].cpu()
intersection_normals = result[3].cpu()
is_intersecting = result[4].cpu()
#sphere_centers = result[5].cpu()
#sphere_radii = result[6].cpu()

is_intersecting = torch.flatten(is_intersecting)
intersections = torch.flatten(intersections, end_dim=2)[is_intersecting].detach().numpy()
intersection_normals = torch.flatten(intersection_normals, end_dim=2)[is_intersecting].detach().numpy()

point_cloud = trimesh.points.PointCloud(intersections, colors=intersection_normals)
scene = trimesh.Scene([point_cloud])
scene.show()
