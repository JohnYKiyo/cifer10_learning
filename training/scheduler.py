from torch.optim.lr_scheduler import CosineAnnealingLR


def get_scheduler(optimizer, **kwargs):
    return CosineAnnealingLR(optimizer, T_max=kwargs.get("T_max", 50))
