import torch
def custom_kernel(data):
    return torch.geqrf(data)
