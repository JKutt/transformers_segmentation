from SimPEG import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pymatsolver import Pardiso as Solver
from SimPEG.electromagnetics.static import resistivity as dc, utils as dcutils
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import norm
import scipy.sparse as sp
import copy
from scipy import stats

import multiprocessing
import time
from scipy.sparse import diags
from pymatsolver import Pardiso as Solver


def perform_rto(simulation, mesh, perturbed_model, initial_x ):

    std = 0.002 * np.abs(simulation.survey.dobs)

    Wd = np.diag(std)

    perturbed_data = np.random.multivariate_normal(simulation.survey.dobs, Wd, size=1)[0, :]
    data_object = data.Data(simulation.survey, dobs=perturbed_data, standard_deviation=std)

    # Define the data misfit. Here the data misfit is the L2 norm of the weighted
    # residual between the observed data and the data predicted for a given model.
    # Within the data misfit, the residual between predicted and observed data are
    # normalized by the data's standard deviation.
    dmis_rto = data_misfit.L1DataMisfit(simulation=simulation, data=data_object)

    # Define the regularization (model objective function)
    reg_rto = regularization.WeightedLeastSquares(
        mesh, alpha_s=1e-4, alpha_x=1, reference_model=perturbed_model
    )
    # reg_rto = regularization.Sparse(
    #     mesh, alpha_s=1, alpha_x=1e-2, reference_model=perturbed_model
    # )

    # now determine best beta
    dmis_eigen = utils.eigenvalue_by_power_iteration(dmis_rto, initial_x)

    reg_eigen = utils.eigenvalue_by_power_iteration(reg_rto, initial_x)

    ratio = np.asarray(dmis_eigen / reg_eigen)
    beta = 1e-6 * ratio

    print(f'beta is: {beta}')

    # Optimization
    opt = optimization.ProjectedGNCG(maxIter=10, upper=np.inf, lower=-np.inf, tolCG=1E-5, maxIterLS=12, )
    opt.remember('xc')

    # Set the inverse problem
    invProb = inverse_problem.BaseInvProblem(dmis_rto,  reg_rto,  opt)
    invProb.startup(initial_x)
    invProb.beta = beta

    return opt.minimize(invProb.evalFunction, initial_x)

def run():

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
    survey1 = dcutils.generate_dcip_survey(
        endl, survey_type="dipole-dipole", dim=mesh.dim,
        a=1, b=1, n=16, d2flag='2.5D'
    )
    survey2 = dcutils.generate_dcip_survey(
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
    simulation = dc.Simulation2DNodal(
        mesh,
        survey,
        sigmaMap=mapping,
        storeJ=True,
        solver=Solver
    )

    std_sim = 0.02

    simulation_data = simulation.make_synthetic_data(mtrue[actcore], relative_error=std_sim, force=True)
    # survey.eps = 1e-4
    simulation.survey.dobs = simulation_data.dobs

    m0 = -np.log(np.median((dcutils.apparent_resistivity_from_voltage(survey, simulation_data.dobs)))) * np.ones(mapping.nP)
    # print(np.median((DCUtils.apparent_resistivity_from_voltage(survey, simulation_data.dobs))))


    # -------------------------------------------------------------------------------------------------

    # rto

    #
    num_samples = 50
    beta_perturb = 1e2
    # Wm = np.sqrt(beta_perturb) * np.eye(mesh.nC)

    n_model_samples = meshCore.nC

    # coefficient matrix
    zero_means_ = np.zeros(1)
    identity_matrix_ = np.eye(1)

    Wm = np.sqrt(beta_perturb) * diags(np.ones(n_model_samples))

    # print('creating s')
    s = np.random.multivariate_normal(zero_means_, identity_matrix_, size=n_model_samples)

    # perturbed_mod = np.linalg.solve(Wm, s3)
    ainv = Solver(Wm)

    perturbed_mod = ainv * s

    results = [None] * num_samples
    draws_beta = [None] * num_samples
    TKO = False

    # process_pool = multiprocessing.pool.ThreadPool(10)
    rto_tasks = []

    print('start processes')

    for ii in range(num_samples):

        # rto_tasks[ii] = process_pool.apply_async(
        #                 perform_rto,
        #                 (
        #                     simulation,
        #                     meshCore,
        #                     perturbed_mod,
        #                     m0,
        #                 )
        #             )
        
        rto_tasks.append(perform_rto(
                            simulation,
                            meshCore,
                            perturbed_mod,
                            m0,
                        ))

    print('finished launch')
    # process_pool.close()
    # process_pool.join()  
    print('getting rto models')
    # for ii in range(10):

    #     results[ii] = rto_tasks[ii].get()

    # recovered_model = np.vstack(results).mean(axis=0)
    np.save(r'C:\Users\johnk\Documents\git\jresearch\rto_models_2d_l1.npy', np.vstack(rto_tasks))


if __name__ == '__main__':
    start = time.time()
    run()
    print(f'completed in: {time.time() - start} seconds')
