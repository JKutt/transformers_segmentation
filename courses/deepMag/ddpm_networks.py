# Import of libraries
import random
import imageio
import numpy as np
from argparse import ArgumentParser
import torch.nn.functional as F
from torch.autograd import grad

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

class UnetEBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super(UnetEBlock, self).__init__()
        self.layerNorm1 = nn.LayerNorm(shape)
        self.layerNorm2 = nn.LayerNorm(shape)
        self.conv1      = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2      = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.conv3      = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x): 
        x1 = self.conv1(x)
        x2 = self.layerNorm1(x1)
        x3 = self.activation(x2)
        x4 = self.conv2(x3)
        x5 = self.layerNorm2(x4)
        x6 = x3 + self.activation(x5)
        x7 = self.conv3(x6)
        out = x6 + self.activation(x7)
        return out



class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(UnetBlock, self).__init__()
        self.layerNorm = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.layerNorm(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

    
class AnIsoDiffBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, nx=32, n_steps=1000, time_emb_dim=100):
        super(AnIsoDiffBlock, self).__init__()
        self.K1 = nn.Parameter(1e-2*torch.randn(out_c, in_c, kernel_size, kernel_size))
        self.K2 = nn.Parameter(1e-2*torch.randn(out_c, out_c, kernel_size, kernel_size))
        self.K3 = nn.Parameter(1e-2*torch.randn(out_c, out_c, kernel_size, kernel_size))
        
        self.B1 = biasNet(nx, out_c)
        self.B2 = biasNet(nx, out_c)
        self.B3 = biasNet(nx, out_c)
        
        
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        
        self.te = self._make_te(time_emb_dim, out_c)
        
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_out, dim_out)
        )

    def act(self, x):
        #x = x/torch.sqrt(torch.mean(x**2) + 1e-3)
        return F.tanh(x)

    def actInt(self, x):
        #x = x/torch.sqrt(torch.mean(x**2) + 1e-3)
        return torch.log(torch.cosh(x))

            
    def forward(self, x, t):
        # e'sigma(K3*(sigma(K2*(sigma(K1*x + B1) + B2) + B3))
        # e'*sigma'(a2)*K2*diag(sigma'(a1))*K1
        #K1'* sigma'(a1)*(K2'*a2)
        def computeEnergy(x,B1, B2, B3):
            a1  = F.conv2d(x, self.K1, padding=self.K1.shape[-1]//2) + B1
            z1  = self.actInt(a1)
            a2  = F.conv2d(z1, self.K2, padding=self.K2.shape[-1]//2) + B2
            z2  = self.actInt(a2)
            a3  = F.conv2d(z2, self.K3, padding=self.K3.shape[-1]//2) + B3
            
            E   = self.actInt(a3).sum() 
            return E, a3, a2, a1
        
        B1  = self.B1(t)
        B2  = self.B2(t)  
        B3  = self.B3(t)  
        
        E, a3, a2, a1 = computeEnergy(x,B1, B2, B3)
        # Backprop
        a3  = self.act(a3)
        a3  = F.conv_transpose2d(a3, self.K3, padding=self.K2.shape[-1]//2)
        a3  = self.act(a2)*a3
        a3  = F.conv_transpose2d(a3, self.K2, padding=self.K2.shape[-1]//2)
        a3  = self.act(a1)*a3
        out  = F.conv_transpose2d(a3, self.K1, padding=self.K1.shape[-1]//2)
        
        return out


class biasNet(nn.Module):
    def __init__(self, nx=32, nhid=32, n_steps=1000, time_emb_dim=100):
        super().__init__()
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        
        self.te1 = self._make_te(time_emb_dim, nhid)
        self.te2 = self._make_te(time_emb_dim, nhid)
        
        self.W1 = nn.Conv2d(nhid, nhid, kernel_size=3, padding=1)
        self.W2 = nn.Conv2d(nhid, nhid, kernel_size=3, padding=1)
        
        self.B  = nn.Parameter(1e-3*torch.rand(1, nhid, nx, nx))
        
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.LeakyReLU(0.2),
        nn.Linear(dim_out, dim_out)
        )

    
    def forward(self, t):
        B = self.B
        t = self.time_embed(t.to(torch.int64)).squeeze()
        T1 = self.te1(t).unsqueeze(-1).unsqueeze(-1)
        T2 = self.te2(t).unsqueeze(-1).unsqueeze(-1)        
        B = F.silu(self.W1(T1*B))
        B = F.silu(self.W2(T2*B))
        
        return B

    
class energyNet(nn.Module):
    def __init__(self, nIn=1, nhid=32, tmax=1000):
        super().__init__()
        
        self.Blk1 = AnIsoDiffBlock(nIn, nhid, nx=32, n_steps=tmax, time_emb_dim=100)
        self.Blk2 = AnIsoDiffBlock(nIn, 2*nhid, nx=16, n_steps=tmax, time_emb_dim=100)
        self.Blk3 = AnIsoDiffBlock(nIn, 4*nhid, nx=8, n_steps=tmax, time_emb_dim=100)
        self.Blk4 = AnIsoDiffBlock(nIn, 8*nhid, nx=4, n_steps=tmax, time_emb_dim=100)

        #self.C1 = nn.Parameter(1e-2*torch.randn(nhid, nIn, 3, 3))
        #self.C2 = nn.Parameter(1e-2*torch.randn(2*nhid, nIn, 3, 3))
        #self.C3 = nn.Parameter(1e-2*torch.randn(4*nhid, nIn, 3, 3))
        #self.C4 = nn.Parameter(1e-2*torch.randn(8*nhid, nIn, 3, 3))
    
    
    def forward(self, x, t):

        
        y1 = self.Blk1(x, t)
        
        x2 = F.interpolate(x, scale_factor=0.5)
        y2 = self.Blk2(x2, t)
        y2 = F.interpolate(y2, scale_factor=2)
        
        x3 = F.interpolate(x, scale_factor=0.25)
        y3 = self.Blk3(x3, t)
        y3 = F.interpolate(y3, scale_factor=4)
        
        x4 = F.interpolate(x, scale_factor=0.125)
        y4 = self.Blk4(x4, t)
        y4 = F.interpolate(y4, scale_factor=8)
        
        
        return y1 + y2 + y3 + y4
        
        
class UNetFlex(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, arch=[1, 16, 32, 64, 128], dims=[32,32]):
        super(UNetFlex, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.DBlocks = nn.ModuleList()
        self.DConvs = nn.ModuleList()
        self.DTE = nn.ModuleList()
        
        # Down blocks
        for i in range(len(arch)-1):
            te = self._make_te(time_emb_dim, arch[i])
            blk = nn.Sequential(
                  UnetBlock((arch[i], dims[0], dims[1]), arch[i], arch[i+1]),
                  UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i+1]),
                  UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i+1]))
        
            down_cnv = nn.Sequential(nn.Conv2d(arch[i+1], arch[i+1], 4, 1, 1),
                                     nn.SiLU(),
                                     nn.Conv2d(arch[i+1], arch[i+1], 3, 2, 1))
            self.DBlocks.append(blk)
            self.DTE.append(te)
            self.DConvs.append(down_cnv)
            dims = [dims[0]//2, dims[1]//2]

        
        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, arch[-1])
        self.blk_mid = nn.Sequential(
            UnetBlock((arch[-1], dims[0], dims[1]), arch[-1], arch[-2]),
            UnetBlock((arch[-2], dims[0], dims[1]), arch[-2],arch[-2]),
            UnetBlock((arch[-2], dims[0], dims[1]), arch[-2], arch[-1])
        )

        self.UBlocks = nn.ModuleList()
        self.UConvs = nn.ModuleList()
        self.UTE = nn.ModuleList()
        # Up cycle
        for i in np.flip(range(len(arch)-1)):


            up = nn.Sequential(
                 nn.ConvTranspose2d(arch[i+1], arch[i+1], 4, 2, 1),
                 nn.SiLU(),
                 nn.ConvTranspose2d(arch[i+1], arch[i+1], 3, 1, 1))

            dims = [dims[0]*2, dims[1]*2]
            teu = self._make_te(time_emb_dim, arch[i+1]*2)
            if i != 0:
                blku = nn.Sequential(
                        UnetBlock((arch[i+1]*2, dims[0], dims[1]), arch[i+1]*2, arch[i+1]),
                        UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i]),
                        UnetBlock((arch[i], dims[0], dims[1]), arch[i], arch[i]))
            else:
                blku = nn.Sequential(
                    UnetBlock((arch[i+1]*2, dims[0], dims[1]), arch[i+1]*2, arch[i+1]),
                    UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i+1]),
                    UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i+1]))
            
            self.UBlocks.append(blku)
            self.UTE.append(teu)
            self.UConvs.append(up)

        self.conv_out = nn.Conv2d(arch[1], arch[0], 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t.to(torch.int64))
        n = len(x)

        # down
        X = [x]
        for i in range(len(self.DBlocks)):

            te = self.DTE[i](t).reshape(n, -1, 1, 1)
            x  = self.DBlocks[i](x + te)
            X.append(x) 
            x  = self.DConvs[i](x)

        x = self.blk_mid(x + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        cnt = -1
        for i in range(len(self.DBlocks)):
            x  = self.UConvs[i](x)
            x  = torch.cat((X[cnt],x), dim=1)  
            te = self.UTE[i](t).reshape(n, -1, 1, 1)
            x = self.UBlocks[i](x + te)  # 
            cnt = cnt-1
        
        out = self.conv_out(x)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    
class energyNetLS(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, arch=[1, 16, 32, 64, 128], dims=[32,32]):
        super(energyNetLS, self).__init__()

        self.Net = UNetFlex(n_steps,time_emb_dim=100, arch=[1, 16, 32, 64, 128], dims=[32,32])

    def forward(self, x, t):
        x = x.clone()
        x.requires_grad = True
        out = self.Net(x, t)
        energy = F.mse_loss(out, x)

        score = grad(energy, x, retain_graph=True, create_graph=True)[0]

        return score