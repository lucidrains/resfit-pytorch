from copy import deepcopy

import torch
from torch.nn import Module, ModuleList, Identity, ParameterList
from torch.func import vmap, functional_call

from ema_pytorch import EMA

from x_mlps_pytorch.normed_mlp import MLP, create_mlp

from einops import pack, unpack

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
        self.critic = critic

        self.actor_ema = EMA(self.actor, beta = actor_ema_decay)
        self.critic_ema = EMA(self.critic, beta = critic_ema_decay)

# critic ensembling
# dealing with the notorious Q overestimation bias, they use an ensembling technique from another paper - train with subset chosen from a population, then actor is optimized with all of the ensemble
# https://arxiv.org/abs/2101.05982

class Ensemble(Module):
    def __init__(
        self,
        net: Module,
        ensemble_size,
        init_std_dev = 2e-2
    ):
        super().__init__()
        self.net = net

        params = dict(net.named_parameters())
        device = next(iter(params.values())).device

        ensemble_params = {name: (torch.randn((ensemble_size, *param.shape), device = device) * init_std_dev).requires_grad_() for name, param in params.items()}

        self.param_names = ensemble_params.keys()
        self.param_values = ParameterList(list(ensemble_params.values()))

        def _forward(params, data):
            return functional_call(net, params, data)

        self.ensemble_forward = vmap(_forward, in_dims = (0, None))

    @property
    def ensemble_params(self):
        return dict(zip(self.param_names, self.param_values))

    def parameters(self):
        return self.ensemble_params.values()

    def forward(
        self,
        data,
        *,
        ids = None,
    ):

        ensemble = self.ensemble_params

        if exists(ids):
            # if `ids` passed in, will forward for only that subset of network
            assert (ids < self.pop_size).all()

            ensemble_params = {key: param[ids] for key, param in pop_params.items()}

        return self.ensemble_forward(dict(ensemble_params), data)

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
