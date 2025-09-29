import torch
from torch.nn import Module, Identity

from ema_pytorch import EMA

from x_mlps_pytorch.ensemble import Ensemble
from x_mlps_pytorch.normed_mlp import MLP, create_mlp

from einops import pack, unpack, einsum

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
            depth = mlp_depth,
            norm_fn = Identity
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

        is_chunked_actions = mlp_input.ndim == 3 # (batch, time, dim)

        if is_chunked_actions:
            mlp_input, packed_batch_time = pack([mlp_input], '* d')

        residual_actions = self.residual_action_mlp(mlp_input)

        if is_chunked_actions:
            residual_actions, = unpack(residual_actions, packed_batch_time, '* d')

        return actions + residual_actions

# td3

class Agent(Module):
    def __init__(
        self,
        actor: ResFitFinetuneWrapper,
        critic: MLP,
        actor_ema_decay = 0.99,
        critic_ema_decay = 0.99
    ):
        super().__init__()

        self.actor = actor

        # critic ensembling
        # dealing with the notorious Q overestimation bias, they use an ensembling technique from another paper - train with subset chosen from a population, then actor is optimized with all of the ensemble
        # https://arxiv.org/abs/2101.05982

        self.critic = Ensemble(critic)

        self.actor_ema = EMA(self.actor, beta = actor_ema_decay)
        self.critic_ema = EMA(self.critic, beta = critic_ema_decay)
