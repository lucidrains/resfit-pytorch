import torch
from torch.nn import Module, ModuleList

from x_mlps_pytorch.normed_mlp

class ResFitFinetuneWrapper(Module):
    def __init__(
        self
    ):
        super().__init__()
