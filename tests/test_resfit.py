import pytest

def test_critic():
    import torch
    from resfit_pytorch.resfit import Ensemble
    from x_mlps_pytorch.normed_mlp import MLP

    critic = MLP(10, 5, 1)
    state = torch.randn(2, 10)

    assert critic(state).shape == (2, 1)
    critics = Ensemble(critic, 10)

    assert critics(state).shape == (10, 2, 1)

    subset_ids = torch.tensor([0, 3, 5])
    assert critics(state, ids = subset_ids).shape == (3, 2, 1)

def test_resfit():
    assert True
