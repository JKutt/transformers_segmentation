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
from sklearn.mixture import GaussianMixture
import copy
from scipy.sparse import diags
# from PGI_DC_example_Utils import plot_pseudoSection, getCylinderPoints

# Python Version
import sys
print(sys.version)

# Reproducible science
seed = 12345
np.random.seed(seed)

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

mtrue = ln_sigback * np.ones(mesh.nC) # + norm(noisemean, noisevar).rvs(mesh.nC)
mprim = copy.deepcopy(mtrue)

csph = (np.sqrt((mesh.gridCC[:, 1] - z0) **
                2. + (mesh.gridCC[:, 0] - x0)**2.)) < r0
mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph])  # + \
    # norm(noisemean, noisevar).rvs(np.prod((mtrue[csph]).shape))

# Define the sphere limit
rsph = (np.sqrt((mesh.gridCC[:, 1] - z1) **
                2. + (mesh.gridCC[:, 0] - x1)**2.)) < r1
mtrue[rsph] = ln_sigr * np.ones_like(mtrue[rsph]) # + \
    # norm(noisemean, noisevar).rvs(np.prod((mtrue[rsph]).shape))

# sphere smaller but higher conductivity
csph = (np.sqrt((mesh.gridCC[:, 1] - z0) **
                2. + (mesh.gridCC[:, 0] - x0)**2.)) < r0
mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph])


mtrue = utils.mkvc(mtrue)
xmin,  xmax = -15., 15
ymin,  ymax = -15., 0.
#xmin,  xmax = mesh.vectorNx.min(), mesh.vectorNx.max()
#ymin,  ymax = mesh.vectorNy.min(), mesh.vectorNy.max()
print(xmin,xmax,ymin,ymax)
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
actcore,  meshCore = discretize.utils.mesh_utils.extract_core_mesh(xyzlim, mesh)
actind = np.ones_like(actcore)

# Setup a Dipole-Dipole Survey with 1m and 2m dipoles
xmin, xmax = -15., 15.
ymin, ymax = 0., 0.
zmin, zmax = 0, 0

endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey1 = DCUtils.generate_dcip_survey(
    endl, survey_type="pole-dipole", dim=mesh.dim,
    a=1, b=1, n=16, d2flag='2.5D'
)
survey2 = DCUtils.generate_dcip_survey(
    endl, survey_type="pole-dipole", dim=mesh.dim,
    a=2, b=2, n=16, d2flag='2.5D'
)
survey3 = DCUtils.generate_dcip_survey(
    endl, survey_type="dipole-pole", dim=mesh.dim,
    a=1, b=1, n=16, d2flag='2.5D'
)
survey4 = DCUtils.generate_dcip_survey(
    endl, survey_type="dipole-pole", dim=mesh.dim,
    a=2, b=2, n=16, d2flag='2.5D'
)

survey = dc.Survey(survey1.source_list + survey2.source_list + survey3.source_list + survey4.source_list)
survey_rto = dc.Survey(survey1.source_list + survey2.source_list + survey3.source_list + survey4.source_list)

# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = maps.ExpMap(mesh)
mapactive = maps.InjectActiveCells(
    mesh=mesh,  indActive=actcore,
    valInactive=-np.log(1e8)
)
mapping = expmap * mapactive

# forward simulation
fwd_simulation = dc.Simulation2DNodal(
    mesh,
    survey,
    sigmaMap=mapping,
    storeJ=True,
    solver=Solver
)


std = 0.02
# survey.dpred(mtrue[actcore])
dpred = fwd_simulation.make_synthetic_data(mtrue[actcore], relative_error=std, force=True)
survey.eps = 1e-4
survey.dobs = dpred.dobs
std = 0.02 * np.abs(dpred.dobs)

simulation = dc.Simulation2DNodal(
    mesh,
    survey,
    sigmaMap=mapping,
    storeJ=True,
    solver=Solver
)

m0 = -np.log(np.median((DCUtils.apparent_resistivity_from_voltage(survey, dpred.dobs)))) * np.ones(mapping.nP)

model_perturb = 1e5
std = 0.01 * np.abs(dpred.dobs)
# Wm = np.sqrt(beta_perturb) * np.eye(mesh.nC)

n_model_samples = meshCore.nC

# coefficient matrix
zero_means_ = np.zeros(1)
identity_matrix_ = np.eye(1)

Wm = np.sqrt(model_perturb) * diags(np.ones(n_model_samples))

# print('creating s')
s = np.random.multivariate_normal(zero_means_, identity_matrix_, size=n_model_samples)

# perturbed_mod = np.linalg.solve(Wm, s3)
ainv = Solver(Wm)

perturbed_mod = ainv * s

perturbed_mod = -np.log(1 / np.exp(m0) + perturbed_mod)

ii = 0
samples_rto = []

while ii < 100:
    Wd = np.diag(std)

    perturbed_data = np.random.multivariate_normal(simulation.survey.dobs, Wd, size=1)[0, :]

    survey_rto.dobs = perturbed_data
    survey_rto.std = np.abs(perturbed_data) * 0.02
    survey_rto.eps = 1e-3

    simulation_rto = dc.Simulation2DNodal(
        mesh,
        survey_rto,
        sigmaMap=mapping,
        storeJ=True,
        solver=Solver
    )

    data_object = data.Data(simulation_rto.survey, dobs=perturbed_data, standard_deviation=survey_rto.std)

    # Data misfit
    survey_rto.eps = 1e-2
    dmis_rto = data_misfit.L2DataMisfit(data=data_object, simulation=simulation_rto)
    dmis_rto.W = 1 / ((0.06 * np.abs(simulation_rto.survey.dobs)) + survey_rto.eps)
    # Regularization
    regmap = maps.IdentityMap(nP=int(actcore.sum()))
    
    # reg = regularization.Sparse(mesh, indActive=active, mapping=regmap)

    # reg_rto = regularization.WeightedLeastSquares(
    #     mesh, 
    #     active_cells=actcore,
    #     mapping=regmap,
    #     reference_model=perturbed_mod
    # )
    # reg_rto.alpha_s = 1/csx**2
    # reg_rto.alpha_x = 100
    # reg_rto.alpha_y = 100
    # reg_rto.alpha_z = 100

    reg_rto = regularization.Sparse(
        mesh, alpha_s=1, active_cells=actcore, mapping=regmap, reference_model=perturbed_mod
    )
    reg_rto.norms = [0, 1, 1]
    

    # Optimization object
    opt_rto = optimization.ProjectedGNCG(maxIter=10, lower=-10, upper=10,
                                    maxIterLS=20, maxIterCG=100, tolCG=1e-5)

    opt_rto.remember('xc')
    opt_rto.printers += [
        optimization.IterationPrinters.phi_s,
        optimization.IterationPrinters.phi_x,
        optimization.IterationPrinters.phi_y,
        optimization.IterationPrinters.phi_z,
    ]
    opt_rto.print_type = "ubc"

    # Set the inverse problem
    invProb_rto = inverse_problem.BaseInvProblem(dmis_rto,  reg_rto,  opt_rto)
    invProb_rto.beta = 1e0

    update_IRLS = directives.Update_IRLS(
        f_min_change=1e-4,
        max_irls_iterations=30,
        coolEpsFact=1.5,
        beta_tol=1e-2,
    )

    # Inversion directives
    Target = directives.TargetMisfit() 
    betaSched = directives.BetaSchedule(coolingFactor=2.,  coolingRate=1.)
    updateSensW = directives.UpdateSensitivityWeights(threshold=1e-2,everyIter=False)
    update_Jacobi = directives.UpdatePreconditioner()

    # we don't add the target directive because we want the IRLS to sample the model space
    inv_rto = inversion.BaseInversion(invProb_rto,  directiveList=[ # updateSensW, 
                                                        # Target,
                                                        betaSched,
                                                        #    update_Jacobi,
                                                        update_IRLS,
                                                        ])

    import time
    start = time.time()
    # Run the inversion
    mopt = inv_rto.run(m0)
    samples_rto.append(mopt)
    print('Inversion took {0} seconds'.format(time.time() - start))

    ii += 1

resulting_samples = np.vstack(samples_rto)
print(resulting_samples.mean(axis=0).shape)
np.save('/home/jkuttai/uq/rto_2d_shperes_l0p1_500samples.npy', resulting_samples)