from drlnd_continuous.model import Actor, Critic
import numpy as np
import random
import copy
import torch
import torch.nn as nn


def test_input_dependence():
    torch.manual_seed(42)
    act = Actor(3, 2, 2, 2)
    out = act.forward(torch.from_numpy(np.array([1., 2., 3.])).float())
    print(out)


def test_understand_model():
    """Playing with some of the ingredients for nn.Module.  In particular the method that adds attributes automatically
    to the output of the parameters method (used for passing all model parameters to the optimizer).  Key is the
    action of the torch.nn.Module.__set_attr__ method which check the type of an attribute and acts accordingly (adding
    to _parameters or _modules dict.  It is then not added to the `__dict__`, but since `__get_attr__` checks these
    dict as well it still acts as if it has been added to `__dict__`.
    """
    t1 = torch.Tensor(2,3)
    assert t1.shape[0] == 2
    assert t1.shape[1] == 3

    class M1(nn.Module):
        def __init__(self):
            super(M1, self).__init__()
            self.layer1 = nn.Linear(10, 10)

    m1 = M1()
    param = list(m1.parameters())
    assert len(param) == 2
    assert param[0].detach().numpy().shape == (10, 10)   # the weights
    assert param[1].detach().numpy().shape == (10,)      # the bias

    class M2(nn.Module):
        def __init__(self):
            super(M2, self).__init__()
            self.layer1 = nn.Linear(10, 10, bias=False)

    m2 = M2()
    param = list(m2.parameters())
    assert len(param) == 1
    assert param[0].detach().numpy().shape == (10, 10)   # the weights

    np = dict(m1.named_parameters())
    print(np)
