from SimPEG import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
from discretize.utils import mkvc, refine_tree_xyz
from SimPEG.electromagnetics import natural_source as ns
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pymatsolver import Pardiso as Solver

from SimPEG.meta import MultiprocessingMetaSimulation

# let import the mesh in UBC format
mesh_file = r"mk2_mesh.msh"
mesh = discretize.TensorMesh.read_UBC(mesh_file)

# load the conductivity model
conductivity_file = r"QAMT_ELLA_Con_Mk2.txt"
model = discretize.TensorMesh.read_model_UBC(mesh, conductivity_file)

# extract the acive files
active = mesh.gridCC[:, 2] < 250.0

model[~active] = 1e-8

# create receivers
rx_x, rx_y = np.meshgrid(np.arange(-250, 1500, 250), np.arange(-250, 2500, 200))
receivers_h = np.hstack((mkvc(rx_x, 2), mkvc(rx_y, 2), 310.0 * np.ones((np.prod(rx_x.shape), 1))))

receivers_e = receivers_h.copy()
receivers_e[:, 2] = 250
print(receivers_h.shape)

# set frequencies
freqs = ['10', '50', '200']

background = 1 / 0.00012 # np.median(model[active])
# create background conductivity model
sigBG = np.zeros(mesh.nC) + 1 / background
sigBG[~active] = 1e-8

# Make a receiver list
rx_list = []
for rx_orientation in ['xy', 'yx']:
    
    rx_list.append(
        
        ns.receivers.PointNaturalSource(
            locations=None,
            locations_e=receivers_e, 
            locations_h=receivers_h,
            orientation=rx_orientation,
            component="real"
        )
    )
    rx_list.append(
        ns.receivers.PointNaturalSource(
            locations=None,
            locations_e=receivers_e, 
            locations_h=receivers_h,
            orientation=rx_orientation, component="imag"
        )
    )

for rx_orientation in ['zx', 'zy']:
    
    rx_list.append(
        
        ns.receivers.Point3DTipper(receivers_h,
                                   orientation=rx_orientation,
                                   component='real'
        )
    )
    rx_list.append(

        ns.receivers.Point3DTipper(receivers_h,
                                   orientation=rx_orientation,
                                   component='imag'
        )
    )

# Source list
src_list = [ns.sources.PlanewaveXYPrimary(rx_list, frequency=float(f)) for f in freqs]

# Survey MT
survey = ns.Survey(src_list)

# Set the mapping
actMap = maps.InjectActiveCells(
    mesh=mesh, indActive=active, valInactive=np.log(1e-8)
)
mapping = maps.ExpMap(mesh) * actMap

# Setup the problem (As a multiprocessing meta sim split by source)
# If you don't want to use the MultiprocessingMetaSim branch, you can just comment
# the below lines out and replace sim with the normal sim that is commented out
# below
mappings = []
sims = []
for src in src_list:
    mappings.append(maps.IdentityMap())
    srv_piece = ns.Survey([src,])
    sims.append(ns.Simulation3DPrimarySecondary(
        mesh, survey=srv_piece, sigmaMap=mapping, sigmaPrimary=sigBG, solver=Solver
    ))

sim = MultiprocessingMetaSimulation(sims, mappings)

sim.model = sigBG[active]

# -------------------------------------------------------------

# calculate the fields

#

import time
source_list = src_list
models = {

    "L_block": np.log(model[active]),
    # "half_space": np.log(sigBG[active])
}

fields = {}

t = time.time()
for key, sig in models.items():
    if key not in fields.keys(): 
        print(f"starting {key}")
        t = time.time()
        fields[key] = sim.fields(sig)
        print(f"done {key}... elapsed time: {time.time()-t:1.1e}s \n")

fwd_data = sim.make_synthetic_data(sig, f=fields["L_block"], add_noise=False)

# Assign uncertainties
fwd_data.relative_error = 0.005  # 5% std
fwd_data.noise_floor = 0.0
# sim.survey.std = np.abs(survey.dobs) * std
# fwd_data.noise_floor[-src_list[-1].nD:] = 0.0005

# Set the conductivity values
sig_half = 0.00012
sig_air = 1e-8
# Make the background model
sigma_0 = np.ones(mesh.nC) * sig_air
sigma_0[active] = sig_half
m_0 = np.log(sigma_0[active])

# Setup the inversion proceedure
# Define a counter
# Data misfit
dmis = data_misfit.L2DataMisfit(data=fwd_data, simulation=sim)

# dmis.W = 1 / (fwd_data.dobs * 0.02 + 1e-6)
# Regularization
regmap = maps.IdentityMap(nP=int(active.sum()))
# reg = regularization.Sparse(mesh, indActive=active, mapping=regmap)

reg = regularization.WeightedLeastSquares(mesh, active_cells=active, mapping=regmap, reference_model=m_0)
reg.alpha_s = 2.5e-5
reg.alpha_x = 1.
reg.alpha_y = 1.
reg.alpha_z = 1


# Optimization
C = utils.Counter()
opt = optimization.ProjectedGNCG(maxIter=5, upper=np.inf, lower=-np.inf, tolCG=1E-2, maxIterCG=20, )
opt.counter = C
opt.remember('xc')

# reg.mrefInSmooth = True
# Inversion problem
invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
invProb.counter = C
# Beta schedule
beta_cool = directives.BetaSchedule(coolingFactor=2, coolingRate=3)
# Initial estimate of beta
beta_est = directives.BetaEstimate_ByEig(beta0_ratio=2e0)
# Target misfit stop
targmis = directives.TargetMisfit()
# targmis.target = survey.nD
saveIter = directives.SaveModelEveryIteration()
# Create an inversion object
directive_list = [beta_est, beta_cool, targmis, saveIter]
inv = inversion.BaseInversion(invProb, directiveList=directive_list)

import time
start = time.time()
# Run the inversion
mopt = inv.run(m_0)
print('Inversion took {0} seconds'.format(time.time() - start))
discretize.TensorMesh.write_model_UBC(mesh=mesh, model=mapping * mopt, file_name='airtipper_and_ground_mt.con')

