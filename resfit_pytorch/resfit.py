import torch
from torch.nn import Module, ModuleList

from ema_pytorch import EMA
from x_mlps_pytorch.normed_mlp import create_mlp

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class ResFitFinetuneWrapper(Module):
    def __init__(
        self,
        base_model: Module,
        dim_action,
        dim_states,
        mlp_depth = 2
    ):
        super().__init__()

        self.base_model = base_model

        self.residual_action_mlp = create_mlp(
            dim_in = dim_action + dim_states,
            dim_out = dim_action,
            depth = mlp_depth
        )

    def parameters(self):
        return self.residual_action_mlp.parameters()

    def forward(
        self,
        states
    ):

        with torch.no_grad():
            self.base_model.eval()
            actions = self.base_model(states)

        mlp_input = torch.cat((states, actions), dim = -1)
        residual_actions = self.residual_action_mlp(mlp_input)

        return actions + residual_actions
