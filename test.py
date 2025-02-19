import torch

print(torch.cuda.get_device_name(torch.cuda.current_device()))
