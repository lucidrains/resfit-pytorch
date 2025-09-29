import torch
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.nn import Module, Identity
from torch.optim import Adam

from ema_pytorch import EMA

from x_mlps_pytorch.ensemble import Ensemble
from x_mlps_pytorch.normed_mlp import MLP, create_mlp

from einops import pack, unpack, einsum

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def choice(num, k):
    return torch.randperm(num)[:k]

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
        num_critics = 10,
        num_critics_for_update = 2,
        num_critic_update_to_actor = 2, # delayed update of the actor - every N critic updates, update the actor
        actor_ema_decay = 0.99,
        critics_ema_decay = 0.99,
        discount_factor = 0.99,
        optim_klass = Adam,
        optim_kwargs: dict = dict(),
        learning_rate = 3e-4
    ):
        super().__init__()

        self.actor = actor

        # critic ensembling
        # dealing with the notorious Q overestimation bias, they use an ensembling technique from another paper - train with subset chosen from a population, then actor is optimized with all of the ensemble
        # https://arxiv.org/abs/2101.05982

        self.critics = Ensemble(critic, num_critics)

        self.num_critics = num_critics
        self.num_critics_for_update = num_critics_for_update # the subselection of critics for learning of the critic
        self.num_critic_update_to_actor = num_critic_update_to_actor

        # exponential smoothing

        self.actor_ema = EMA(self.actor, beta = actor_ema_decay)
        self.critics_ema = EMA(self.critic, beta = critic_ema_decay)

        # optimizers

        self.actor_optim = optim_klass(self.actor.parameters(), lr = learning_rate)
        self.critics_optim = optim_klass(self.critics.parameters(), lr = learning_rate)

        # step - for keeping track of when actors update

        self.register_buffer('step', tensor(1))

    def train_step(
        self,
        state,
        actions,
        rewards,
        next_state,
        next_actions,
        terminal
    ):
        step = self.step.item()
        self.step.add_(1)

        # update critic

        critic_indices = choice(self.num_critics, self.num_critics_for_update)

        pred_q_value = self.critics(state, actions, ids = critic_indices)

        next_q_value = self.critics_ema(next_state, next_actions, ids = critic_indices)

        # bandaid over the overestimation issue
        # take minimum of Q value

        pred_q_value, next_q_value = tuple(reduce(t, 'critics ... -> ...', 'min') for t in (pred_q_value, next_q_value))

        # bellman's

        target_q_value = rewards * (~terminal).float() * self.discount_factor * next_q_value

        critics_loss = F.mse_loss(pred_q_value, target_q_value)

        critics_loss.backward()

        self.critics_optim.step()
        self.critics_optim.zero_grad()

        self.critics_ema.update()

        # early return if actor is not to be updated

        if not divisible_by(step, self.num_critic_update_to_actor):
            return

        # actor is updated on all of the critics, not a subset

        action = self.actor(state)

        q_value = self.critics_ema.forward_eval(state, action)

        # gradient ascent

        (-q_value).mean().backward()

        self.actor_optim.step()
        self.actor_optim.zero_grad()

        self.actor_ema.update()

    def forward(
        self,
        dataset: Dataset
    ):
        raise NotImplementedError
