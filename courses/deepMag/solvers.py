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
from scipy.sparse.linalg import spsolve

class CGLS(nn.Module):
    def __init__(self, forOp, reg=None, CGLSit=100, eps = 1e-2, device='cuda'):
        super(CGLS, self).__init__()
        self.forOp = forOp
        self.reg = reg
        self.nCGLSiter = CGLSit
        self.eps = eps

    def forward(self, b, xref):

        x = xref
        beta = torch.tensor([1e-1])
        
        d_diff = b - (self.forOp(x)*1e8)
        r_d = 0.5 * torch.dot(d_diff.flatten(), d_diff.flatten())
        r_reg = self.reg(x)
        dims = xref.shape
        print(f"dmis: {r_d} m_mis: {r_reg}")
        r = r_d + beta * r_reg
        if r.norm()/b.norm()<self.eps:
                return x, r
        # print(r_d.shape,  self.reg.deriv(x).shape, self.forOp.adjoint(d_diff).flatten().shape)
        s = (self.forOp.adjoint(d_diff) *1e8) + beta * self.reg.deriv(x).view(dims)
        
        # Initialize
        p      = s
        norms0 = torch.norm(s)
        gamma  = norms0**2

        for k in range(self.nCGLSiter):

            # q_d = b - 
            
            q = b - (self.forOp(p) * 1e8)
            delta = torch.norm(q)**2  + beta * self.reg(p)
            alpha = gamma / delta
            print(f"alpha, p: {alpha}, {p}")
            x     = x + alpha*p
            r     = r - alpha*q

            print(k, r.norm().item()/b.norm().item())
            if r.norm()/b.norm()<self.eps:
                return x, r
       
            s = (self.forOp.adjoint(r)*1e8) + beta * self.reg.deriv(x).view(dims)
        
            norms  = torch.norm(s)
            gamma1 = gamma
            gamma  = norms**2
            beta   = gamma / gamma1
            p      = s + beta*p
     
        return x, r

