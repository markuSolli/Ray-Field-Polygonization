import torch
from ifield.models import intersection_fields

checkpoint = 'MARF/experiments/logdir/tensorboard/experiment-stanfordv12-bunny-both2marf-16atom-50xinscr-10dmiss-geom-25cnrml-8x512fc-leaky_relu-hit-0minatomstdngxp-500sphgrow-10mdrop-layernorm-multi_view-10dmv-nocond-100cwu500clr70tvs-2023-05-31-0010-nqzh/checkpoints/epoch=197-step=13860.ckpt'

model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(checkpoint)
model.eval()

# (w, h, 3) and (3) dtype=float32
origins = torch.Tensor([0, 0, -1]).to(torch.float32)
dirs = torch.Tensor([[[0, 0, 1]]]).to(torch.float32)

intersections = model.forward(dict(origins=origins, dirs=dirs))
print(intersections)