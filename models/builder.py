import torch.nn as nn

from . import simple

def BuildAutoEncoder(model: str) -> nn.Module:
    if model == 'simple':
        model = simple.SimpleAutoencoder()
    else:
        raise NotImplementedError("Model not implemented")

    return model
