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


def perform_rto(
    dobs_org,
    perturbed_model,
    initial_x,
    idx,
    csx:float=5.0,
    csy:float=5.0,
    csz:float=5.0,
    ncx:int=163,
    ncz:int=61,
    npad:int=12
):
    """
        method to run a sampling of rto

        csx: cell size x
        type: float
        csx: cell size y
        type: float
        csx: cell size z
        type: float
        ncx: cell size x
        type: int
        ncz: cell size z
        type: int
        ncy: cell size y
        type: int
    
    """

    # create mesh

    # Vectors of cell lengthts in each direction
    hx = [(csx, npad,  -1.5), (csx, ncx), (csx, npad,  1.5)]
    hz = [(csz, npad, -1.5), (csz, ncz)]
    # Create mesh
    mesh = discretize.TensorMesh([hx,  hz], x0="CN")
    mesh.x0[1] = mesh.x0[1] + csz / 2.

    xmin, xmax = -400., 400.
    ymin, ymax = -300., 0.
    zmin, zmax = 0, 0
    xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
    actcore,  meshCore = utils.mesh_utils.ExtractCoreMesh(xyzlim, mesh)

    # ------------------------------------------------------------------------------------------------

    # generate acquisition data (true data)

    #

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

    survey.dobs = dobs_org

    # create the simulation
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

    std = 0.002 * np.abs(simulation.survey.dobs)

    Wd = np.diag(std)

    perturbed_data = np.random.multivariate_normal(simulation.survey.dobs, Wd, size=1)[0, :]

    data_object = data.Data(simulation.survey, dobs=perturbed_data, standard_deviation=std)

    # Define the data misfit. Here the data misfit is the L2 norm of the weighted
    # residual between the observed data and the data predicted for a given model.
    # Within the data misfit, the residual between predicted and observed data are
    # normalized by the data's standard deviation.
    dmis_rto = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)

    # Define the regularization (model objective function)
    reg_rto = regularization.WeightedLeastSquares(
        meshCore, alpha_s=1e-4, alpha_x=1, reference_model=perturbed_model
    )
    # reg_rto = regularization.Sparse(
    #     mesh, alpha_s=1, reference_model=perturbed_model
    # )

    # now determine best beta
    dmis_eigen = utils.eigenvalue_by_power_iteration(dmis_rto, initial_x)

    reg_eigen = utils.eigenvalue_by_power_iteration(reg_rto, initial_x)

    ratio = np.asarray(dmis_eigen / reg_eigen)
    beta = 1e-8 * ratio

    print(f'beta is: {beta}')

    # Optimization
    opt = optimization.ProjectedGNCG(maxIter=10, upper=np.inf, lower=-np.inf, tolCG=1E-5, maxIterLS=12, )
    opt.remember('xc')

    # Set the inverse problem
    invProb = inverse_problem.BaseInvProblem(dmis_rto,  reg_rto,  opt)
    invProb.startup(initial_x)
    invProb.beta = beta

    sample_model = opt.minimize(invProb.evalFunction, initial_x)

    np.save(f'/home/juan/Documents/git/jresearch/bayesian_inversion/rtobe_samples/sample_{idx}.npy', sample_model)

    return sample_model

def run():

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

    # Dike 45*
    dike0 = mesh.gridCC[:,1] > fault_function(mesh.gridCC[:,0],1, 100)
    dike1 = mesh.gridCC[:,1] < fault_function(mesh.gridCC[:,0],1, 175)
    dike = np.logical_and(dike0,dike1)

    model[dike]=4

    # plot
    fig,ax = plt.subplots(3, 1,figsize=(10,20))
    mm1 = mesh.plotImage(model, ax=ax[0], pcolorOpts={'cmap':'Spectral_r'})

    # define conductivities
    res_true = np.ones(mesh.nC)
    res_true[model==3]= 500.0
    res_true[model==4]= 10.0

    cond_true = 1./res_true

    mtrue = np.log(cond_true)

    xmin, xmax = -400., 400.
    ymin, ymax = -300., 0.
    zmin, zmax = 0, 0
    xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
    actcore,  meshCore = utils.mesh_utils.ExtractCoreMesh(xyzlim, mesh)
    actind = np.ones_like(actcore)

    # ------------------------------------------------------------------------------------------------

    # generate acquisition data (true data)

    #

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
    num_samples = 500
    beta_perturb = 1e4
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

    # process_pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    rto_tasks = [None] * num_samples

    print('start processes')
    start = time.time()

    for ii in range(num_samples):

        print(f"\n\n performing sampling: {ii}\n\n")

        # rto_tasks[ii] = process_pool.apply_async(
        #                 perform_rto,
        #                 (
        #                     simulation_data.dobs,
        #                     perturbed_mod,
        #                     m0,
        #                 )
        #             )
        
        rto_tasks[ii] = perform_rto(
                            simulation_data.dobs,
                            perturbed_mod,
                            m0,
                            ii
                        )

    print(f'finished launch: {time.time() - start} seconds')
    # process_pool.close()
    # process_pool.join()  
    # print('getting rto models')
    # for ii in range(num_samples):

    #     results[ii] = rto_tasks[ii].get()
    results = rto_tasks
    # recovered_model = np.vstack(results).mean(axis=0)
    np.save(r'./rto_models_2d_dip_parallel.npy', np.vstack(results))


if __name__ == '__main__':
    start = time.time()
    run()
    print(f'completed in: {time.time() - start} seconds')
