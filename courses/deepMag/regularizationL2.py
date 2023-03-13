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


class L2_regularization(nn.Module):
    def __init__(
            
            self,
            mesh: tmesh , 
            alpha_s: float=1.0,
            alpha_x=1,
            alpha_y=1,
            alpha_z=1,
            alpha_xx=0.0,
            alpha_yy=0.0,
            alpha_zz=0.0,
            length_scale_x=None,
            length_scale_y=None,
            length_scale_z=None,
            mapping=None,
            reference_model=None,
            reference_model_in_smooth=False,
            weights=None,
            device='cuda'
    ):
        super(L2_regularization, self).__init__()
        self.mesh = mesh
        self.reference_model = reference_model
        self._cell_gradient_x = None
        self._cell_gradient_y = None
        self._cell_gradient_z = None
        self.mapping = mapping
        self.W = weights
        self.device = device

        # initiate weights
        if weights is None:

            W = torch.ones(mesh.number_of_cells, 1)

        self.Wx = torch.sparse.spdiags(W.T, torch.tensor([0]), (mesh.number_of_cells, torch.prod(torch.tensor(mesh.shape_faces_x())))).coalesce()
        self.Wy = torch.sparse.spdiags(W.T, torch.tensor([0]), (mesh.number_of_cells, torch.prod(torch.tensor(mesh.shape_faces_y())))).coalesce()
        self.Wz = torch.sparse.spdiags(W.T, torch.tensor([0]), (mesh.number_of_cells, torch.prod(torch.tensor(mesh.shape_faces_z())))).coalesce()

    
    def forward(self, model):
        """

            Evaulate the model misfit
        
        """

        model = model.flatten()

        dm = self.delta_m(model)

        f_m_x = self.Wx @ self.cell_gradient_x() @ dm
        f_m_y = self.Wy @ self.cell_gradient_y() @ dm
        f_m_z = self.Wz @ self.cell_gradient_z() @ dm

        r = f_m_x + f_m_y + f_m_z

        return 0.5 * r.dot(r)

    
    def deriv(self, model, v=None):

        model = model.flatten()

        dm = self.delta_m(model)

        f_m_x = self.Wx @ self.cell_gradient_x() @ dm
        f_m_y = self.Wy @ self.cell_gradient_y() @ dm
        f_m_z = self.Wz @ self.cell_gradient_z() @ dm

        f_m_d_m = torch.sparse.spdiags(torch.ones(self.mesh.number_of_cells, 1).T, torch.tensor([0]), (self.mesh.number_of_cells, self.mesh.number_of_cells)).coalesce()

        f_m_deriv_x = self.cell_gradient_x() @ f_m_d_m
        f_m_deriv_y = self.cell_gradient_y() @ f_m_d_m
        f_m_deriv_z = self.cell_gradient_z() @ f_m_d_m
    
        r_x = f_m_x
        r_y = f_m_y
        r_z = f_m_z
        
        return f_m_deriv_x.T @ (self.Wx.T @ r_x) + f_m_deriv_y.T @ (self.Wy.T @ r_y) + f_m_deriv_z.T @ (self.Wz.T @ r_z)
    
    
    def delta_m(self, model):

        return model - self.reference_model

    
    def Pac(self) -> torch.sparse_coo_tensor:
        """
        
            Projection matrix that takes from the reduced space of active cells to
            full modelling space (ie. nC x nactive_cells).
        
        """

        nC = self.mesh.number_of_cells
            
        return torch.sparse.spdiags(torch.ones([nC, 1]).T, torch.tensor([0]), (nC, nC)).coalesce()
    

    def Pafx(self) -> torch.sparse_coo_tensor:
        """
            
            Projection matrix that takes from the reduced space of active x-faces
            to full modelling space (ie. nFx x nactive_cells_Fx )
        
        """



        nFx = torch.prod(torch.tensor(self.mesh.shape_faces_x()))
    

        self._Pafx = torch.sparse.spdiags(torch.ones([nFx, 1]).T, torch.tensor([0]), (nFx, nFx)).coalesce()

        return self._Pafx
    
    
    def Pafy(self) -> torch.sparse_coo_tensor:
        """
            
            Projection matrix that takes from the reduced space of active x-faces
            to full modelling space (ie. nFy x nactive_cells_Fy )
        
        """


        nFy = torch.prod(torch.tensor(self.mesh.shape_faces_y()))
    

        self._Pafy = torch.sparse.spdiags(torch.ones([nFy, 1]).T, torch.tensor([0]), (nFy, nFy)).coalesce()

        return self._Pafy
    
    
    def Pafz(self) -> torch.sparse_coo_tensor:
        """
            
            Projection matrix that takes from the reduced space of active x-faces
            to full modelling space (ie. nFz x nactive_cells_Fz )
        
        """


        nFz = torch.prod(torch.tensor(self.mesh.shape_faces_z()))

        self._Pafz = torch.sparse.spdiags(torch.ones([nFz, 1]).T, torch.tensor([0]), (nFz, nFz)).coalesce()

        return self._Pafz

    
    def cell_gradient(self) -> torch.sparse_coo_tensor:
        """
        
            Cell centered gradient matrix
        
        """

        return torch.vstack([self.cell_gradient_x(), self.cell_gradient_y(), self.cell_gradient_z()])
            

    def cell_gradient_x(self) -> torch.sparse_coo_tensor:
        """
        
            Cell centered gradient matrix in the x-direction.
        
        """

        self._cell_gradient_x = (
                    self.Pafx().T @ self.mesh.cell_gradient_x() @ self.Pac()
                )
        
        return self._cell_gradient_x

    
    def cell_gradient_y(self) -> torch.sparse_coo_tensor:
        """
        
            Cell centered gradient matrix in the y-direction.
        
        """

        self._cell_gradient_y = (
                    self.Pafy().T @ self.mesh.cell_gradient_y() @ self.Pac()
                )
        
        return self._cell_gradient_y


    def cell_gradient_z(self) -> torch.sparse_coo_tensor:
        """
        
            Cell centered gradient matrix in the z-direction.
        
        """

        self._cell_gradient_z = (
                    self.Pafz().T @ self.mesh.cell_gradient_z() @ self.Pac()
                )
        
        return self._cell_gradient_z
    