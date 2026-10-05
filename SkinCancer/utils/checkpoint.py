import os

import torch


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device, strict=False):
    if not os.path.exists(path):
        return False
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=strict)
    return True
