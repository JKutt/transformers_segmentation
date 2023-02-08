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
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from time import time

class pytorch_mesh():

    """

        a light version of discretize that uses pytorch as base of all mechanics
        behind the scenes

    """
    
    def __init__(
        self,
        mesh_size: list=[],
        cell_size: list=[],
        padding_growth: list=[],
        number_of_padding_cells: int=5,
    ) -> None:
        """

            initiate a pytorch mesh with parameters defining the geometry.

        """

        self.dim              = torch.FloatTensor(mesh_size)
        self.h                = torch.FloatTensor(cell_size)
        self.padding          = padding_growth
        self.number_pad_cells = number_of_padding_cells



class magnetics(nn.Module):
    
    def __init__(self, dim, h, dirs, device='cuda'):
        super(magnetics, self).__init__()
        self.dim    = dim   # Mesh size [nx, ny, nz]
        self.h      = h     # cell size [Dx, Dy, Dz]
        self.dirs   = dirs  # magnetic field directions [A, I, A0, I0]
        self.device = device 
        dV = torch.prod(self.h)
        mu_0 = 4*np.pi*1e-7
        zeta = mu_0 / (4 * np.pi)
        self.mudV = zeta*dV

    def fft_kernel(self, P, center):
            # use centers and the response and shift the data for FD operations
            S = torch.fft.fftshift(torch.roll(P, shifts=center, dims=[0,1]))
            # take the fft
            S = torch.fft.fft2(S)
            # shift again to swap quadrants
            S = torch.fft.fftshift(S)
            return S

    def forward(self, M):
        """
            Solve the forward problem using FFT
            :param M: model
            :type M: Tensor
        """
        
        # define the constants
        Dz = self.h[2]
        Z  = Dz/2
        
        Data = 0 
        
        # loop through each layer of the model
        for i in range(M.shape[-1]):
            
            # pull out the layer from the model
            I = M[:,:,i]

            # calculate the response the layer of the model
            P, center, Rf = self.psfLayer(Z)
            
            # use centers and the response and shift the data for FD operations
            S = self.fft_kernel(P, center)

            # do the same to model tensor
            I_fft = torch.fft.fftshift(I)
            I_fft = torch.fft.fft2(I_fft)
            I_fft = torch.fft.fftshift(I_fft)

            # perform the FD operations
            B = S * I_fft
            B = torch.fft.fftshift(B)

            # convert back to spatial domain
            B = torch.real(torch.fft.ifft2(B))

            # add the data response from the layer
            Data = Data+B
            Z = Z + Dz
        
        return self.mudV * Data
    
    def adjoint_a(self, I):
        # Adjoint for testing
        a = torch.ones(self.dim[0], self.dim[1], self.dim[2], requires_grad=True, device=self.device)
        d = self.forward(a)
        b = torch.sum(d*I)
        out = grad(b,a)[0]
        return out

    def adjoint(self, I):
        
        Dz = self.h[2]
        
        Z = Dz/2
        M  = torch.zeros(self.dim[0], self.dim[1], self.dim[2], device=self.device)
        
        for i in range(M.shape[-1]):

            # calculate the response the layer of the model
            P, center, Rf = self.psfLayer(Z)
            # use centers and the response and shift the data for FD operations
            S = self.fft_kernel(P, center)

            # do the same to model tensor
            I_fft = torch.fft.fft2(I)
            I_fft = torch.fft.fftshift(I_fft)

            # perform the FD operations
            B = torch.adjoint(S) * I_fft

            # convert back to spatial domain
            B = torch.fft.fftshift(B)
            B = torch.real(torch.fft.ifft2(B))
            B = torch.fft.fftshift(B)
            
            # add the data response from the layer
            M[:,:,i] = B
            Z = Z + Dz
        
        return self.mudV*M
    
    def psfLayer(self, Z):
         #I is the magnetization dip angle 
         # A is the magnetization deflection angle
         # I0 is the geomagnetic dip angle
         # A0 is the geomagnetic deflection angle

        dim2 = torch.div(self.dim,2,rounding_mode='floor')
        Dx = self.h[0]
        Dy = self.h[1]
        I  = self.dirs[0]
        A  = self.dirs[1]
        I0 = self.dirs[2]
        A0 = self.dirs[3]
        
        x = Dx*torch.arange(-dim2[0]+1,dim2[0]+1, device=self.device)
        y = Dy*torch.arange(-dim2[1]+1,dim2[1]+1, device=self.device)
        X,Y = torch.meshgrid(x,y)

        # Get center ready for fftshift.
        center = [1 - int(dim2[0]), 1 - int(dim2[1])]

        Rf = torch.sqrt(X**2 + Y**2 + Z**2)**5
        PSFx = (2*X**2 - Y**2 - Z**2)/Rf*torch.cos(I)*torch.sin(A) + \
               3*X*Y/Rf*torch.cos(I)*torch.cos(A) + \
               3*X*Z/Rf*torch.sin(I)

        PSFy = 3*X*Y/Rf*torch.cos(I)*torch.sin(A) + \
               (2*Y**2 - X**2 - Z**2)/Rf*torch.cos(I)*torch.cos(A) + \
               3*Y*Z/Rf*torch.sin(I)

        PSFz = 3*X*Z/Rf*torch.cos(I)*torch.sin(A) + \
               3*Z*Y/Rf*torch.cos(I)*torch.cos(A) +\
               (2*Z**2 - X**2 - Y**2)/Rf*torch.sin(I)

        PSF = PSFx*torch.cos(I0)*torch.cos(A0) + \
              PSFy*torch.cos(I0)*torch.sin(A0) + \
              PSFz*torch.sin(I0) 
        
        return PSF, center, Rf

if True:

    #time_s = time()
    
    # Adjoint test
    adjoint_test = True
    
    dim = torch.tensor([1024,1024,512])
    h = torch.tensor([100.0, 100.0, 100.0])
    dirs = torch.tensor([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
    forMod = magnetics(dim, h, dirs)

    # set the magnetization model
    M = torch.ones(dim[0], dim[1], dim[2], device='cuda') * 0.0
    # M[600:800, 600:800, 100:400] = 0.1
    M[400:600, 400:600, 100:400] = 0.1

    D = forMod(M)

    #print(f'Done: {time() - time_s} seconds')

    plt.imshow(D.view(1024, 1024).cpu().detach().numpy(), cmap='rainbow')
    # plt.imshow(M[:, 512, :].view(1024, 512).T.cpu().detach().numpy(), cmap='rainbow')
    plt.title('Centerd Block I=90 D=90 degrees')
    plt.colorbar()
    plt.show()

    # Adjoint test
    if adjoint_test:
        dim = torch.tensor([1024,1024,64])
        h = torch.tensor([100.0, 100.0, 100.0])
        dirs = torch.tensor([np.pi/4, np.pi/4, np.pi/4, np.pi/4])
        forMod = magnetics(dim, h, dirs)

        M = torch.rand(dim[0], dim[1], dim[2], device='cuda')
        D = forMod(M)
        Q = torch.rand_like(D)
        W = forMod.adjoint(Q)
        Wa = forMod.adjoint_a(Q)

        print('Adjoint test 1', torch.sum(M*W).item(), torch.sum(D*Q).item())
        print('Adjoint test 2', (W-Wa).norm()/Wa.norm())
        

    print('Done')
