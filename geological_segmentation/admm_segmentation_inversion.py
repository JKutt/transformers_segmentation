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
    
    def __init__(self, dmisfit, aug_lag, regularize, gamma=1.0e-1):
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
        self.regularize = regularize
        self.beta = 1e0

    def __call__(self, model, return_g=True, return_H=True):
        f = self.dmisfit.simulation.fields(model)

        # Data misfit term
        phi_d = self.dmisfit(model, f=f)

        # Regularization term
        phi_m = reg(model)
        phi = phi_d + self.gamma * phi_m
        out = (phi,)
        print(f'phi_d: {phi_d}, phi_m: {phi_m}')

        if return_g:

            phi_dDeriv = self.dmisfit.deriv(model, f=f)
            phi_mDeriv = self.aug_lag.deriv(model)
            phi_rmDeriv = self.regularize.deriv(model)

            g = phi_dDeriv + self.gamma * phi_mDeriv + self.beta * phi_rmDeriv
            out += (g,)
        
        if return_H:
            def H_fun(v):
                phi_d2Deriv = self.dmisfit.deriv2(model, v, f=f)
                phi_m2Deriv = self.aug_lag.deriv2(model, v=v)
                phi_rm2Deriv = self.regularize.deriv2(model, v=v)

                return phi_d2Deriv + self.gamma * phi_m2Deriv + self.beta * phi_rm2Deriv

            H = sp.linalg.LinearOperator((model.size, model.size), H_fun, dtype=model.dtype)
            out += (H,)

        return out if len(out) > 1 else out[0]

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
    mesh.gridCC[:,0], np.tan(30 * np.pi / 180), -175)
dike1 = mesh.gridCC[:,1] < fault_function(
    mesh.gridCC[:,0], np.tan(30 * np.pi / 180), -100)
dike = np.logical_and(dike0,dike1)

model[dike]=4

# plot
# fig,ax = plt.subplots(3, 1,figsize=(10,20))
# mm1 = mesh.plotImage(model, ax=ax[0], pcolorOpts={'cmap':'Spectral_r'})

# ax[0].set_xlim([-1000,1000])
# ax[0].set_ylim([-250,0])
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
# mm = meshCore.plot_image(
    
#     1/(cond_true)[actcore],
#     ax=ax[0],
#     pcolorOpts={'cmap':'Spectral_r'}

# )

# utils.plot2Ddata(

#     meshCore.gridCC,mtrue[actcore],nx=500,ny=500,
#     contourOpts={'alpha':0},
#     #clim=[0,5],
#     ax=ax[0],
#     level=True,
#     ncontour=2,
#     levelOpts={'colors':'k','linewidths':2,'linestyles':'--'},
#     method='nearest'
    
# )
#plt.gca().set_ylim([-200,0])
# ax[0].set_aspect(1)
# plt.colorbar(mm[0], label=r'$\Omega$ m')
# ax[0].set_title('True model')

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
    noise_floor=5e-3,
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
# dmis.W = 1./((dc_data.dobs*0.05) + np.quantile(np.abs(dc_data.dobs), 0.07))

m0 = np.log(1/500.0) * np.ones(mapping.nP) # 1/dcutils.apparent_resistivity_from_voltage(survey, dc_data.dobs).mean()) * np.ones(mapping.nP)
z0 = m0.copy()
u0 = np.zeros_like(z0) # np.random.randn(z0.shape[0])
idenMap = maps.IdentityMap(nP=m0.shape[0])

m = m0.copy()
z = z0.copy()
u = u0.copy()

opt = optimization.ProjectedGNCG(maxIter=1, upper=np.inf, lower=np.log(1/600), tolCG=1E-5, maxIterLS=20, )

opt.remember('xc')

solver_opts = dmis.simulation.solver_opts

reg = regularization.Smallness(
    mesh=meshCore,
    reference_model=(z + u),
)

# # Weighting
reg_mean = regularization.WeightedLeastSquares(
    mesh, 
    active_cells=actcore,
    mapping=idenMap,
    reference_model=m0
)

reg_mean.alpha_s = 0
reg_mean.alpha_x = 100
reg_mean.alpha_y = 100

opt.bfgsH0 = Solver(
    sp.csr_matrix(reg.deriv2(m0) + reg_mean.deriv2(m0)), **solver_opts
)

# segmentor = geoseg.SamClassificationModel(
#     meshCore,
#     segmentation_model_checkpoint=r"/home/juanito/Documents/trained_models/sam_vit_h_4b8939.pth"
# )

eval_funct = evaluate_objective(dmis, reg, reg_mean)
trade_off = 1e-3
for ii in range(10):
   
    reg.reference_model=(z + u)
    print(f'm update {ii}')
    m = opt.minimize(evaluate_objective(dmis, reg, reg_mean, gamma=trade_off), m)
    print(f'z update {ii}')
    z = soft_thresholding(m + u, trade_off)

    # # new z
    # segmentor.fit(m)

    # z = segmentor.predict(m)

    u = u + (m - z)

    np.save(f'model_{ii}.npy', m)
    np.save(f'z_variabe_{ii}.npy', z)

    if ii > 5:
       trade_off = 1e-1

fig, ax = plt.subplots(3, 1, figsize=(10, 5))

meshCore.plot_image(1/ np.exp(z), ax=ax[0], pcolor_opts={'cmap':'Spectral'}, clim=[10, 500])
meshCore.plot_image(1/ np.exp(m), ax=ax[1], pcolor_opts={'cmap':'Spectral'}, clim=[10, 500])
ax[0].axis('equal')
ax[1].axis('equal')
ax[2].hist(1/ np.exp(z), 100, label='z')
ax[2].hist(1/ np.exp(m), 100, label='m', alpha=0.4)

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

utils.plot2Ddata(

    meshCore.gridCC,mtrue[actcore],nx=500,ny=500,
    contourOpts={'alpha':0},
    #clim=[0,5],
    ax=ax[1],
    level=True,
    ncontour=2,
    levelOpts={'colors':'k','linewidths':2,'linestyles':'--'},
    method='nearest'
    
)

plt.show()

np.save('model.npy', m)
np.save('z_variabe.npy', z)
