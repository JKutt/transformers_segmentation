# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:05:31 2023

@author: Rout
"""

import os, sys
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import scipy as sp
from scipy.constants import mu_0
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from scipy.sparse.linalg import spsolve

import torchvision
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from scipy.ndimage import zoom
import copy

from magneticsForward import magnetics
from networks import UNet3D
from torchModelDataLoader import modelDataset

class inverseNet(nn.Module):
    
    def __init__(self, net, forMat, shape):
 
        super().__init__()
        
        self.forMat = forMat
        self.net = net
        self.niter = 4    # layers
        self.shape = shape

    def forward(self, data):
        
        x = torch.zeros(self.shape[0], self.shape[1],self.shape[2], self.shape[3], self.shape[4], device=data.device)
        for i in (range(self.niter)):
            t = torch.tensor([i]).to(torch.float32).to(data.device)
            
            # data fitting step
            r = data - (self.forMat(x))
            g = self.forMat.adjoint(r)
            Ag = self.forMat(g) 
            alpha = (r*Ag).mean()/(Ag*Ag).mean()
            x = x + alpha*g
            # breakpoint()
            '''
            g.mean()            tensor(-4.5864e-15)
            g.max()            tensor(7.3216e-07)
            g.min()            tensor(-4.7488e-08)
            g.std()            tensor(3.1670e-08)
            alpha            tensor(6.1715e+11)
            
            for mu_0 = 1
            g.min()            tensor(-0.0019)
            g.max()            tensor(0.0675)
            g.std()            tensor(0.0029)
            g.mean()            tensor(0.0002)
            alpha            tensor(0.9832)
            '''
            
            #breakpoint()
            # correction step
            x = self.net(x)
            #x = self.net(x[None,None,:,:,:])[0,0,:,:,:]#, t)

        return x
    
class cnet(nn.Module):
    
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, X, t):    # t : iteration step 
        return X
    
def loss_fn(x, forMat, inverseNet):
    
    data = forMat(x)
    
    xrec = inverseNet(data)
    # breakpoint()
    loss = F.mse_loss(xrec, x)
    
    return loss

# import matplotlib.pyplot as plt

batch_size = 4
dataset = modelDataset(directory="./models", dims=3, scale=1)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# dataloader_ref = DataLoader(dataset, batch_size=50000, shuffle=True)

# for j, data in tqdm(enumerate(dataloader_ref, 0)):
  #  print("data min: ", data.min())
   # print("data max: ", data.max())
   # print("data mean: ", data.mean())
   # print("data std:", data.std())

'''
data min:  tensor(0.)
data max:  tensor(0.9912)
data mean:  tensor(0.0087)
data std: tensor(0.0370)
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

shape = torch.tensor([batch_size, 1, 32, 32, 16])
net = UNet3D(in_channels=1, num_classes=1, level_channels=[64, 128, 256], bottleneck_channel=512)
# net = cnet(shape)
if torch.cuda.is_available():
    net = net.cuda()

# summary(model=net, input_size=(1, 32, 32, 16), batch_size=-1)
optimizer = optim.Adam(params=net.parameters(),lr=0.001)

h = torch.tensor([100.0, 100.0, 100.0])
dirs = torch.tensor([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
forMat = magnetics(dim=shape, h=h, dirs=dirs, device=device)

inet = inverseNet(net, forMat, shape)

# x = torch.zeros(shape[0], shape[1], shape[2])
# x[10:16, 10:16, 4:7] = 1.0

niterations = 1000    # epoch
hh = torch.zeros(niterations)
bestloss = 1e11
# breakpoint()

for i in tqdm(range(niterations)):
    
    mean_loss = 0
    
    for j, data in tqdm(enumerate(dataloader, 0)):   
    # torch.Size([100, 128, 128, 64]) tensor(1.0505) tensor(1.0000e-06) tensor(0.0149) tensor(0.0646)
        data = (data - 0.0087)/0.0370
        x = zoom(data, (1, 1, 0.25, 0.25,0.25))
        x = torch.Tensor(x.copy())
        
        optimizer.zero_grad()
        
        loss = loss_fn(x.to(device), forMat, inet)          
        loss.backward()
        optimizer.step()
        
        mean_loss = mean_loss + loss.item()
        
    hh[i] = mean_loss/j
    if i%5 == 0:
        print('iter = %3d   loss = %3e   best_loss = %3e'%(i, mean_loss, bestloss))
    if mean_loss<bestloss:
        bestloss = mean_loss
        bestnet = copy.deepcopy(net)
        print('iter = %3d       best loss = %3e'%(i, bestloss))

torch.save(bestnet.state_dict(),'net3GNoddyBatch_4iterlr1e2.pt')
np.savetxt("epoch_loss_4iterlr1e2.csv", hh.detach().cpu().numpy(), delimiter=",")
print('done training')

xhat = inet(d).detach().cpu().numpy()

print('shape of d = ', d.shape)

f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(x[:, :, 15])
plt.colorbar()
f.add_subplot(1,2, 2)
plt.imshow(xhat[:, :, 15])
plt.colorbar()
plt.show()
plt.savefig("comparison_4iterlr1e2.png")
