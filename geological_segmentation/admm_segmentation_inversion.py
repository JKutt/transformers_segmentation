import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm

import geological_segmentation as geoseg
from SimPEG import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pymatsolver import Pardiso as Solver
from SimPEG.electromagnetics.static import resistivity as dc, utils as dcutils
import scipy.sparse as sp

# -------------------------------------------------------------------------------------------------
def steepest_descent(sim, data, niter):

  r = data.copy()
  g = sim.Jtvec(r, v=None)
  u = 0
  for i in range(niter):

    Ag = sim.dpred(g)
    mu = (Ag*r).mean()/(Ag*Ag).mean()
    u  = u + mu*g
    r  = r - mu*Ag
    g  = sim.Jtvec(r, v=None)
    print('%3d       %3.2e'%(i, r.norm()/data.norm()))
  return u

def conjugate_gradient(forProb, reg, alpha, d, niter=10, tol=1e-3):

  def HmatVec(x, forProb, reg, alpha):
    Ax = forProb(x)
    ATAx = forProb.adjoint(Ax)
    _, WTWx = reg(x)
    Hx = ATAx + alpha*(WTWx).view(-1)
    return Hx

  rhs = forProb.adjoint(d)
  u   = 0
  r   = rhs.clone()

  p   = r.clone()
  for i in range(niter):
    Hp = HmatVec(p, forProb, reg, alpha)
    rsq = (r*r).mean()
    mu = (r*r).mean()/(p*Hp).mean()
    u = u + mu*p
    r = r - mu*Hp

    if r.norm()/rhs.norm() < tol:
      return u

    beta = (r*r).mean()/rsq
    p    = r + beta*p
    misfit = (forProb(u) - d).norm()/d.norm()
    print('%3d      %3.2e      %3.2e'%(i, r.norm()/rhs.norm(), misfit))
  return u


# -------------------------------------------------------------------------------------------------

# create a 2d mesh for a dc simulation

#

#2D mesh
csx,  csy,  csz = 5.,  5.,  5.
# Number of core cells in each direction
ncx,  ncz = 163,  61
# Number of padding cells to add in each direction
npad = 12
# Vectors of cell lengthts in each direction
hx = [(csx, npad,  -1.5), (csx, ncx), (csx, npad,  1.5)]
hz = [(csz, npad, -1.5), (csz, ncz)]
# Create mesh
mesh = discretize.TensorMesh([hx,  hz], x0="CN")
mesh.x0[1] = mesh.x0[1] + csz / 2.

print(mesh)

# -----------------------------------------------------------------------

# create a synthetic model for a dc simulation

#

model = 3 * np.ones(mesh.nC, dtype='int64')

# divide domain by  45* fault at 100 m
fault_function = lambda x, slope, shift: slope * x + shift

# Dike 30*
dike0 = mesh.gridCC[:,1] > fault_function(
    mesh.gridCC[:,0], np.tan(30 * np.pi / 180), -75)
dike1 = mesh.gridCC[:,1] < fault_function(
    mesh.gridCC[:,0], np.tan(30 * np.pi / 180), 0)
dike = np.logical_and(dike0,dike1)

model[dike]=4

# plot
fig,ax = plt.subplots(3, 1,figsize=(10,20))
mm1 = mesh.plotImage(model, ax=ax[0], pcolorOpts={'cmap':'Spectral_r'})

ax[0].set_xlim([-1000,1000])
ax[0].set_ylim([-250,0])
# ax[0].set_aspect(2)
# plt.colorbar(mm1[0])


# define conductivities
res_true = np.ones(mesh.nC)
res_true[model==3]= 500
res_true[model==4]= 10

index_deep = mesh.gridCC[:, 1] >= -50

index_deeper = mesh.gridCC[:, 1] < -200

res_true[index_deep] = 500
res_true[index_deeper] = 500

cond_true = 1./res_true

mtrue = np.log(cond_true)

xmin, xmax = -400., 400.
ymin, ymax = -300., 0.
zmin, zmax = 0, 0
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
actcore,  meshCore = utils.mesh_utils.extract_core_mesh(xyzlim, mesh)
actind = np.ones_like(actcore)

# plot
mm = meshCore.plot_image(
    
    1/(cond_true)[actcore],
    ax=ax[0],
    pcolorOpts={'cmap':'Spectral_r'}

)

utils.plot2Ddata(

    meshCore.gridCC,mtrue[actcore],nx=500,ny=500,
    contourOpts={'alpha':0},
    #clim=[0,5],
    ax=ax[0],
    level=True,
    ncontour=2,
    levelOpts={'colors':'k','linewidths':2,'linestyles':'--'},
    method='nearest'
    
)
#plt.gca().set_ylim([-200,0])
ax[0].set_aspect(1)
plt.colorbar(mm[0], label=r'$\Omega$ m')
ax[0].set_title('True model')

xmin, xmax = -350., 350.
ymin, ymax = 0., 0.
zmin, zmax = 0, 0

endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
srclist = []

for dipole in np.linspace(25,250,10):
    
    survey1 = dcutils.generate_dcip_survey(
        
        endl, survey_type="pole-dipole",
        dim=mesh.dim,
        a=dipole,
        b=dipole,
        n=16,
    
    )

    # print(dipole)

    survey2 = dcutils.generate_dcip_survey(
        
        endl, survey_type="dipole-pole",
        dim=mesh.dim,
        a=dipole,
        b=dipole,
        n=16,
    
    )
    
    srclist +=(survey1.source_list)
    srclist +=(survey2.source_list)

survey = dc.Survey(srclist)

# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = maps.ExpMap(mesh)
mapactive = maps.InjectActiveCells(
    
    mesh=mesh,
    indActive=actcore,
    valInactive=-np.log(100)

)
mapping = expmap * mapactive
simulation = dc.Simulation2DNodal(
    
    mesh, 
    survey=survey, 
    sigmaMap=mapping,
    solver=Solver,
    nky=8

)

# -----------------------------------------------------------------------

# create synthetic data and view psuedo-section

#

relative_measurement_error = 0.01
dc_data = simulation.make_synthetic_data(
    
    mtrue[actcore],
    relative_error=relative_measurement_error,
    noise_floor=6e-3,
    force=True,
    add_noise=True,

)

# dc_data.noise_floor = np.quantile(np.abs(dc_data.dobs), 0.1)

relative_error_list = (np.abs(dc_data.standard_deviation/dc_data.dobs))
print(relative_error_list.min())
print(relative_error_list.max())


# uhat = steepest_descent(simulation.Jtvec, dc_data.dobs, 20)

# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# meshCore.plot_image(uhat, ax=ax[0], pcolorOpts={'cmap':'Spectral_r'})
# plt.show()

# -----------------------------------------------------------------------

dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)

m0 = np.log(1/dcutils.apparent_resistivity_from_voltage(survey, dc_data.dobs).mean()) * np.ones(mapping.nP)
z0 = m0.copy()
u0 = np.zeros_like(z0)
idenMap = maps.IdentityMap(nP=m0.shape[0])

reg = regularization.Smallness(
    mesh=meshCore,
    reference_model=(z0 + u0),
)

def soft_thresholding(x, threshold):
    """
    Apply soft thresholding to the elements of a vector.

    Parameters:
        x (array-like): Input vector.
        threshold (float): Threshold value.

    Returns:
        array-like: Soft thresholded vector.
    """
    # Soft thresholding operation
    return np.sign(x) * np.maximum(0, np.abs(x) - threshold)

class evaluate_objective():
    
    def __init__(self, dmisfit, aug_lag, gamma=1.0):
        """
        Evaluate the objective function for the inversion problem.

        Parameters:
            m (array-like): Model vector.
            dmis (data_misfit.DataMisfit): Data misfit object.
            reg (regularization.Regularization): Regularization object.

        Returns:
            float: Objective function value.
        """

        self.gamma = gamma
        self.aug_lag = aug_lag
        self.dmisfit = dmisfit

    def __call__(self, m, return_g=True, return_H=True):
        f = self.dmisfit.simulation.fields(m)

        # Data misfit term
        phi_d = self.dmisfit(m, f=f)

        print(f'phi_d: {phi_d}')

        # Regularization term
        phi_m = reg(m)

        phi_dDeriv = self.dmisfit.deriv(m, f=f)
        phi_mDeriv = self.aug_lag.deriv(m)

        g = phi_dDeriv + self.gamma * phi_mDeriv

        def H_fun(v):
            phi_d2Deriv = self.dmisfit.deriv2(m, v, f=f)
            phi_m2Deriv = self.aug_lag.deriv2(m, v=v)

            return phi_d2Deriv + self.gamma * phi_m2Deriv

        H = sp.linalg.LinearOperator((m.size, m.size), H_fun, dtype=m.dtype)

        return phi_d + phi_m, g, H

m = m0.copy()
z = z0.copy()
u = u0.copy()

opt = optimization.ProjectedGNCG(maxIter=60, upper=np.inf, lower=-np.inf, tolCG=1E-5, maxIterLS=20, )
opt.Solver = Solver
opt.remember('xc')

for ii in range(10):
   
    reg = regularization.Smallness(
        mesh=meshCore,
        reference_model=(z + u),
    )

    # eval_funct = evaluate_objective(dmis, reg)

    m = opt.minimize(evaluate_objective(dmis, reg), m)

    z = soft_thresholding(m + u, 1.0)

    u = u + (m - z)