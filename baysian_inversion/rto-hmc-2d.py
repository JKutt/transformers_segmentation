from SimPEG import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pymatsolver import Pardiso as Solver
from SimPEG.electromagnetics.static import resistivity as dc, utils as DCUtils
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import norm
import scipy.sparse as sp
import copy
from scipy import stats

def hmc(dmisfit, m_misfit, n_samples, x0, n_leap_frog=4 ,step_size=0.25, p_std=1):

    samples = [x0, ]
    M = np.diag(p_std**2 * np.ones_like(x0))
    momentum_dist = stats.multivariate_normal(mean=np.zeros_like(x0), cov=M)
    M_inv = 1/p_std**2 * np.ones_like(x0)
    
    n_accept = 0 
    deriv =dmisfit.deriv(samples[0]) + 1e-3 * m_misfit.deriv(samples[0])
    x0_nlp = dmisfit(samples[0]) + 1e-3 * m_misfit(samples[0])
    for i in range(n_samples):
        x0 = samples[-1]
        x1 = x0.copy()
        
        p0 = momentum_dist.rvs()
        p1 = p0.copy()

        # leapfrog integration begin
        deriv = dmisfit.deriv(x1) + 1e-3 * m_misfit.deriv(x1)

        for s in range(n_leap_frog):
            p1 -= step_size * deriv/2
            x1 += step_size * M_inv * p1
            deriv = dmisfit.deriv(x1) + 1e-3 * m_misfit.deriv(x1)
            p1 -= step_size * deriv/2
        # leapfrog integration end
        p1 *= -1
        # reverse momentum at the final location for reversibility
        # In this case the momentum distribution is symmetric, so it doesn't matter...

        #metropolis acceptance
        x1_nlp = dmisfit(x1) + 1e-3 * m_misfit(x1)
        
        p0_nlp = -momentum_dist.logpdf(p0)
        p1_nlp = -momentum_dist.logpdf(p1)
        
        # Account for negatives AND log(probabiltiies)...
        target = x0_nlp - x1_nlp # f(x1)/f(x0)
        adjustment = p0_nlp - p1_nlp # g(p1)/g(p0)
        acceptance = target + adjustment # rf * rg
        
        if np.log(np.random.random()) <= acceptance:
            samples.append(x1)
            x0_nlp = x1_nlp
            n_accept += 1
        else:
            samples.append(x0)

    return np.array(samples), n_accept

# 2D Mesh
#########
csx,  csy,  csz = 0.25,  0.25,  0.25
# Number of core cells in each direction
ncx,  ncz = 123,  61
# Number of padding cells to add in each direction
npad = 12
# Vectors of cell lengthts in each direction
hx = [(csx, npad,  -1.5), (csx, ncx), (csx, npad,  1.5)]
hz = [(csz, npad, -1.5), (csz, ncz)]
# Create mesh
mesh = discretize.TensorMesh([hx,  hz], x0="CN")
mesh.x0[1] = mesh.x0[1] + csz / 2.

# 2-cylinders Model Creation
##########################
# Spheres parameters
x0,  z0,  r0 = -6.,  -5.,  3.
x1,  z1,  r1 = 6.,  -5.,  3.

ln_sigback = -np.log(500.)
ln_sigc = -np.log(90.)
ln_sigr = -np.log(50.)

# Add some variability to the physical property model
noisemean = 0.
noisevar = np.sqrt(0.001)
ln_over = -2.

mtrue = ln_sigback * np.ones(mesh.nC) + norm(noisemean, noisevar).rvs(mesh.nC)
mprim = copy.deepcopy(mtrue)

csph = (np.sqrt((mesh.gridCC[:, 1] - z0) **
                2. + (mesh.gridCC[:, 0] - x0)**2.)) < r0
mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph]) + \
    norm(noisemean, noisevar).rvs(np.prod((mtrue[csph]).shape))

# Define the sphere limit
rsph = (np.sqrt((mesh.gridCC[:, 1] - z1) **
                2. + (mesh.gridCC[:, 0] - x1)**2.)) < r1
mtrue[rsph] = ln_sigr * np.ones_like(mtrue[rsph]) + \
    norm(noisemean, noisevar).rvs(np.prod((mtrue[rsph]).shape))

# sphere smaller but higher conductivity
csph = (np.sqrt((mesh.gridCC[:, 1] - z0) **
                2. + (mesh.gridCC[:, 0] - x0)**2.)) < r0
mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph]) + \
    norm(noisemean, noisevar).rvs(np.prod((mtrue[csph]).shape))


mtrue = utils.mkvc(mtrue)
xmin,  xmax = -15., 15
ymin,  ymax = -15., 0.
#xmin,  xmax = mesh.vectorNx.min(), mesh.vectorNx.max()
#ymin,  ymax = mesh.vectorNy.min(), mesh.vectorNy.max()
print(xmin,xmax,ymin,ymax)
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
actcore,  meshCore = discretize.utils.mesh_utils.extract_core_mesh(xyzlim, mesh)
actind = np.ones_like(actcore)

clim = [mtrue.min(), mtrue.max()]

fig, ax = plt.subplots(1,1,figsize=(10,5))
dat = meshCore.plotImage(mtrue[actcore], ax=ax, clim=clim, pcolorOpts={'cmap':"Spectral"})
ax.set_title('Tikhonov inversion',fontsize=24)
ax.set_aspect('equal')
ax.set_ylim([-15,0])
ax.set_xlabel('x (m)',fontsize=22)
ax.set_ylabel('z (m)',fontsize=22)
ax.tick_params(labelsize=20)
fig.subplots_adjust(right=0.85)
plt.colorbar(dat[0])
plt.show()

# ------------------------------------------------------------------------------------------------

# generate acquisition data (true data)

#

# Setup a Dipole-Dipole Survey with 1m and 2m dipoles
xmin, xmax = -15., 15.
ymin, ymax = 0., 0.
zmin, zmax = 0, 0

endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey1 = DCUtils.generate_dcip_survey(
    endl, survey_type="dipole-dipole", dim=mesh.dim,
    a=1, b=1, n=16, d2flag='2.5D'
)
survey2 = DCUtils.generate_dcip_survey(
    endl, survey_type="dipole-dipole", dim=mesh.dim,
    a=2, b=2, n=16, d2flag='2.5D'
)

survey = dc.Survey(survey1.source_list + survey2.source_list)

# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = maps.ExpMap(mesh)
mapactive = maps.InjectActiveCells(
    mesh=mesh,  indActive=actcore,
    valInactive=-np.log(1e8)
)
mapping = expmap * mapactive
sim = dc.Simulation2DNodal(
    mesh,
    survey,
    sigmaMap=mapping,
    storeJ=True,
    solver=Solver
)

std_sim = 0.02

simulation_data = sim.make_synthetic_data(mtrue[actcore], relative_error=std_sim, force=True)
# survey.eps = 1e-4

m0 = -np.log(np.median((DCUtils.apparent_resistivity_from_voltage(survey, simulation_data.dobs)))) * np.ones(mapping.nP)
print(np.median((DCUtils.apparent_resistivity_from_voltage(survey, simulation_data.dobs))))

# ------------------------------------------------------------------------------------------------

# setup hmc

#
starting_model = m0

std = 0.02 * np.abs(simulation_data.dobs)
n_model_samples = 26

data_object_hmc = data.Data(survey, dobs = simulation_data.dobs.copy(), standard_deviation=std)

dmis_hmc = data_misfit.L2DataMisfit(simulation=sim, data=data_object_hmc)

# Define the regularization (model objective function)
reg_hmc = regularization.WeightedLeastSquares(
    meshCore, alpha_s=1e-4, alpha_x=1,
)


# # functions for eval and g to feed hmc
# # Example usage with the function f(x) = x^2 - 4x + 4
# def objective_func(x):
#     # print(dmis(x).shape, beta_init, reg(x).shape)
#     r = dmis_hmc(x) + beta_init * reg_hmc(x)
#     return r

# def gradient_func(x):
#     return dmis_hmc.deriv(x) + beta_init * reg_hmc.deriv(x)

samples, n_accept = hmc(
    dmis_hmc, reg_hmc, n_samples=1000, x0=starting_model, n_leap_frog=10 ,step_size=0.0002, p_std=0.04
)

print(f"number of accepted samples: {n_accept}")

# result.append(rto_model)
recovered_conductivity_model = np.median(np.vstack(samples), axis=0)

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_axes([0.15, 0.15, 0.67, 0.75])
mesh.plot_slice(
    plotting_map * recovered_conductivity_model_log10,
    ax=ax1,
    normal="Y",
    ind=int(len(mesh.h[1]) / 2),
    # grid=True,
    clim=(true_conductivity_model_log10.min(), true_conductivity_model_log10.max()),
    pcolor_opts={"cmap": 'Spectral'},
)
ax1.set_title("Recovered Model - LHS Sampled injections Bayesian inversion")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-1000, 1000])
ax1.set_ylim([-1000, 0])

plt.show()

