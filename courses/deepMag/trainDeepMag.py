import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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

import magneticsForward as MF
from regularizationL2 import L2_regularization
from torch_maps import IdentityMap
import torch_mesh as tmesh
import solvers

dim = torch.tensor([128,128,64])
h = torch.tensor([100.0, 100.0, 100.0])
dirs = torch.tensor([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
forMod = MF.magnetics(dim, h, dirs, device='cpu')
#forMod = MF.testFM(dim, h, dirs, device='cpu')

n1 = 128
n2 = 128
n3 = 64
h1 = 100 * torch.ones(n1)
h2 = 100 * torch.ones(n2)
h3 = 100 * torch.ones(n3)

jmesh = tmesh.torch_mesh(h1, h2, h3)
mapping = IdentityMap(mesh=jmesh, nP=jmesh.number_of_cells)

# set the magnetization model
M = torch.ones(dim[0], dim[1], dim[2], device='cpu') * 0.0
M[40:60, 40:60, 10:40] = 0.1

D = forMod(M)*1e8
reg = L2_regularization(jmesh, mapping=mapping, reference_model=M*0)

sol = solvers.CGLS(forMod, reg=reg, CGLSit=100, eps = 1e-5, device='cpu')

x, r = sol(D, M*0)

# Ax = b   A Ez = b 
print('Done: ', x[:, :, 32].shape)
plt.imshow(M[:, :, 32].T.detach().numpy())
plt.show()
