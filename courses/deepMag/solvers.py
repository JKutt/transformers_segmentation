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
    def __init__(self, forOp, CGLSit=100, eps = 1e-2, device='cuda'):
        super(CGLS, self).__init__()
        self.forOp = forOp
        self.nCGLSiter = CGLSit
        self.eps = eps

    def forward(self, b, xref):

        x = xref
        
        r = b - self.forOp(x)
        if r.norm()/b.norm()<self.eps:
                return x, r
        s = self.forOp.adjoint(r)
        
        # Initialize
        p      = s
        norms0 = torch.norm(s)
        gamma  = norms0**2

        for k in range(self.nCGLSiter):
    
            q = self.forOp(p) 
            delta = torch.norm(q)**2
            alpha = gamma / delta
    
            x     = x + alpha*p
            r     = r - alpha*q

            print(k, r.norm().item()/b.norm().item())
            if r.norm()/b.norm()<self.eps:
                return x, r
       
            s = self.forOp.adjoint(r)
        
            norms  = torch.norm(s)
            gamma1 = gamma
            gamma  = norms**2
            beta   = gamma / gamma1
            p      = s + beta*p
     
        return x, r
