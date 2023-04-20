import os, sys
import torch
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
import torch_mesh as tmesh

class ObjectiveFun(nn.Module):

    def __init__(self, function_list=[]) -> None:
        
        super(ObjectiveFun, self).__init__()
        self.functions = function_list

    def forward(self, x):

        output = []

        for fun in self.functions:
            g = fun(x)
            print(f" g: {g.shape}")
            if len(g.shape) > 1:
                g = g.flatten() * 1e8
            output += [g]

        return torch.hstack(output)
    
    def adjoint(self, m):



        output = []

        for fun in self.functions:

            h = fun.adjoint(m)

            if len(h.shape) > 1:
                h = h.flatten()

            output += [h]

        return torch.hstack(output)

