import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import unittest
from scipy.constants import mu_0
from discretize.tests import check_derivative
import discretize
import matplotlib.patheffects as pe
from SimPEG import dask
from SimPEG.electromagnetics import natural_source as nsem
from SimPEG.electromagnetics.static import utils as sutils
from SimPEG import (
    maps, utils, optimization, objective_function, inversion, inverse_problem, directives,
    data_misfit, regularization, data
)
from discretize import TensorMesh
from pymatsolver import Pardiso
import gmsh
from scipy.spatial import cKDTree
from scipy.stats import norm
from dask.distributed import Client, LocalCluster
import time


def create_tile_em_misfit(
    sources,
    obs,
    uncert,
    global_mesh,
    global_active,
    tile_id, mstart, 
    use_global=False):
    """

        Used to create frequency domain tile of simulation Simulation2DMagneticField 

    """

    local_survey = nsem.Survey(sources)

    electrodes = sources[0].receiver_list
    local_survey.dobs = obs
    local_survey.std = uncert

    # Create tile map between global and local
    if use_global:
        local_mesh = global_mesh
        local_active = global_active
        local_map = maps.IdentityMap(nP=int(local_active.sum()))
    else:
        local_mesh = create_nested_mesh(
            electrodes, global_mesh,
        )
        local_map = maps.TileMap(global_mesh, global_active, local_mesh)
        local_active = local_map.local_active

    actmap = maps.InjectActiveCells(
        local_mesh, indActive=local_active, valInactive=np.log(1e-8)
    )
    expmap = maps.ExpMap(local_mesh)
    mapping = expmap * actmap
    # Create the local misfit
    max_chunk_size = 256

    simulation = nsem.simulation.Simulation2DMagneticField(
        local_mesh,
        survey=local_survey,
        sigmaMap=mapping,
        solver=Pardiso,
    )

    simulation.sensitivity_path = './Sensitivity/Tile' + str(tile_id) + '/'
    # print(simulation.getSourceTerm().shape)
    data_object = data.Data(
        local_survey,
        dobs=obs,
        standard_deviation=uncert,
    )
    data_object.dobs = obs
    data_object.standard_deviation = uncert
    local_misfit = data_misfit.L2DataMisfit(
        data=data_object, simulation=simulation, model_map=local_map
    )
    local_misfit.W = 1 / uncert

    return local_misfit


def get_surface_weights(mesh, actind, values, octree=False):
    """
        Function that determines the surface weights to be applied
        either an octree or tensor mesh
        input: discretize mesh object: mesh
               numpy array: active domain cells
               numpy array: value of weights at each cell
                            at depth determined by size of array
    """
    n_layer = values.size

    uniqXYlocs, topoCC = sutils.gettopoCC(mesh, actind, option='center')

    tree = cKDTree(mesh.gridCC)

    if octree:
        d, inds = tree.query(np.c_[uniqXYlocs, topoCC])

    else:
        d, inds = tree.query(np.c_[uniqXYlocs.gridCC, topoCC])

    # Regularization (just for mesh use)
    regmap = maps.IdentityMap(nP=int(actind.sum()))

    reg = regularization.Sparse(
        mesh, indActive=actind,
        mapping=regmap
    )

    surface_weights_temp = np.ones(mesh.nC)
    surface_weights_temp[inds] = values[0]
    surface_weights = surface_weights_temp.copy()

    if n_layer > 1:

        for i in range(n_layer - 1):
            temp = np.zeros(mesh.nC)

            temp[actind] = reg.regmesh.aveFy2CC * reg.regmesh.cellDiffyStencil * surface_weights_temp[actind]

            inds = temp == 0.5

            surface_weights[inds] = values[i + 1]

    return surface_weights


def run():
    """

        Runs a 2D MT inversion on synethic buried sphere with apparent resistivity and phase

    """
    tc = time.time()
    deriv_type = "sigma"
    sim_type = "h"
    fixed_boundary=True

    cluster = LocalCluster(processes=False)
    client = Client(cluster)
        
    print('[INFO] creating Tensor Mesh...')
    mesh = discretize.TensorMesh(
    [
        #[(min cell size,left padding cells, growth factor),(min cell size, amount of cells @ that size),(min cell size,right padding cells, growth factor)]
        [(200,1),(100,1),(75,1),(50,1),(25,1),(20,1),(15,1),(12.5,146),(15,1),(20,1),(25,1),(50,1),(75,1),(100,1),(200,1)],
        [(187.5,1),(100,1),(75,1),(50,1),(37.5,2),(25,3),(18.75,5),(12.5,5),(10,5),(8.75,10),(7.5,15),(6.25,12)]
    ], x0=[-485, 2580])

    mesh.plotGrid()

    b = 1000
    A = 200

    # create topography

    Z = A * np.exp(-0.5 * ((mesh.vectorCCx / b) ** 2.0 )) + 3370

    topo = np.vstack([mesh.vectorCCx, Z]).T


    plt.plot(mesh.vectorCCx, Z, 'r')

    print('[NOTE] Active cells are being generated')
    actinds = utils.surface2ind_topo(mesh, topo, method='linear')       # active indicies
    print('[NOTE] Active cells completed')


    # create the synthetic model
    sigma_back = 1e-3
    sigma_right = 1e-1
    sigma_porph = 1
    sigma_basement = 1e-3
    sigma_air = 1e-8

    #  Add some variability to the physical property model
    noisemean = 0.
    noisevar = np.sqrt(0.0001)
    ln_over = -2.

    cells = mesh.cell_centers
    sigma = np.ones(mesh.n_cells) * sigma_back
    
    # Conductive sphere
    x0 = 700
    z0 = 3300
    r0 = 100
    csph = (
        np.sqrt(
            (mesh.gridCC[:, 0] - x0) ** 2.0
            + (mesh.gridCC[:, 1] - z0) ** 2.0
        )
    ) < r0

    sigma[csph] = sigma_porph
    sigma[~actinds] = sigma_air



    print(actinds.sum(), actinds.shape, topo.shape, sigma.shape)

    actmap = maps.InjectActiveCells(

        mesh, indActive=actinds, valInactive=np.log(1e-8)

    )

    if deriv_type == "sigma":
        sim_kwargs = {"sigmaMap": maps.ExpMap() * actmap}
        test_mod = np.log(sigma)
    else:
        sim_kwargs = {"muMap": maps.ExpMap(), "sigma": sigma}
        test_mod = np.log(mu_0) * np.ones(mesh.n_cells)

    frequencies = np.logspace(-1, 4, 20)

    z_flight_height = A * np.exp(-0.5 * ((np.linspace(0, 1700, 35) / b) ** 2.0 )) + 3370


    rx_locs = np.c_[np.linspace(0, 1700, 35), z_flight_height]

    plt.plot(rx_locs[:, 0], rx_locs[:, 1], 'g')

    print(frequencies, rx_locs.shape)


    # if fixed_boundary:

    #     actmap = maps.InjectActiveCells(

    #         mesh, indActive=actinds, valInactive=np.log(1e-8)

    #     )

    #     # get field from 1D simulation
    #     survey_1d = nsem.Survey(
    #         [nsem.sources.Planewave([], frequency=f) for f in frequencies]
    #     )
    #     mesh_1d = TensorMesh([mesh.h[1]], [mesh.origin[1]])
    #     sim_1d = nsem.simulation.Simulation1DMagneticField(
    #         mesh_1d, survey=survey_1d, sigmaMap=maps.IdentityMap()
    #     )

    #     b_left, b_right, _, __ = mesh.cell_boundary_indices
    #     f_left = sim_1d.fields(sigma[b_left])
    #     f_right = sim_1d.fields(sigma[b_right])

    #     b_e = mesh.boundary_edges
    #     top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
    #     left = np.where(b_e[:, 0] == mesh.nodes_x[0])
    #     right = np.where(b_e[:, 0] == mesh.nodes_x[-1])
    #     e_bc = {}
    #     for src in survey_1d.source_list:
    #         e_bc_freq = np.zeros(mesh.boundary_edges.shape[0], dtype=np.complex)
    #         e_bc_freq[top] = 1.0
    #         e_bc_freq[right] = f_right[src, "e"][:, 0]
    #         e_bc_freq[left] = f_left[src, "e"][:, 0]
    #         e_bc[src.frequency] = e_bc_freq
    #     sim_kwargs["e_bc"] = e_bc

    rx_list = [
    #     nsem.receivers.PointNaturalSource(
    #         rx_locs, orientation="yx", component="real"
    #     ),
    #     nsem.receivers.PointNaturalSource(
    #         rx_locs, orientation="yx", component="imag"
    #     ),
    #     nsem.receivers.Point3DTipper(
    #         rx_locs, orientation="zx", component="imag"
    #     ),
        nsem.receivers.PointNaturalSource(
            rx_locs, orientation="yx", component="apparent_resistivity"
        ),
        nsem.receivers.PointNaturalSource(
            rx_locs, orientation="yx", component="phase"
        ),
    ]

    src_list = [nsem.sources.Planewave(rx_list, frequency=f) for f in frequencies]

    survey = nsem.Survey(src_list)

    sim = nsem.simulation.Simulation2DMagneticField(
        mesh,
        survey=survey,
        **sim_kwargs,
        solver=Pardiso,
    )

    data_obs = sim.dpred(np.log(sigma[actinds])).compute()

    data_obs = data_obs + (np.random.randn(data_obs.shape[0]) * (data_obs * 0.05))

    plt.show()


    # -----------------------------------------------------------------------------
    
    # Plot synthetic model

    #

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    mtrue = sigma  # np.ones(mesh.nC) * 1 / np.median(rho_app)
    mtrue[~actinds] = np.nan
    clim = [0, 2000]
    print(mtrue.shape, actinds.sum(), actinds.shape, topo.shape, sigma.shape, mesh.nC)

    dat = mesh.plot_image((1 / (mtrue)), ax=ax, grid=False, clim=clim,
    #                      gridOpts={'alpha': 0.2},
                        pcolorOpts={"cmap": "rainbow_r"}
                        )
    ax.set_title('Resistivity')
    cb = plt.colorbar(dat[0], ax=ax)
    ax.set_aspect('equal')

    plt.show()


    # ---------------------------------------------------------------------------------

    # Assigning errors

    #

    # assign error
    data_rs = np.reshape(data_obs, (frequencies.shape[0], rx_locs.shape[0] * 2))
    rs_error = np.zeros(data_rs.shape)
    print(rs_error.shape)

    # set some extended use variables
    num_sites = rx_locs.shape[0]

    eps = 1

    eps = np.percentile(1 / data_rs[:, num_sites:], 10)

    data_for_tile = {}
    std_for_tile = {}
    print(eps)
    for ii in range(data_rs.shape[0]):
        
        
    #     print(eps_i)
        std_r = 0.1
        std_i = 0.1
        
        rs_error[ii, :num_sites] = 1 / (np.abs(1 / data_rs[ii, :num_sites]) * std_r + eps)
        rs_error[ii, num_sites:] = np.abs(data_rs[ii, num_sites:]) * std_i + 40

        print(f"\n\n good errot\n\n { data_rs[ii, :]}")
        data_for_tile[str(frequencies[ii])] = data_rs[ii, :]
        std_for_tile[str(frequencies[ii])] = rs_error[ii, :]
        
        plt.plot(rx_locs[:, 0], data_rs[ii, :num_sites], '-o')
        plt.plot(rx_locs[:, 0], data_rs[ii, num_sites:], '-or')
        plt.title(str(frequencies[ii]))
    plt.show()
        
    std = rs_error.flatten('F')

    # plt.loglog(frequencies, data_obs[:20], '.')
    plt.plot(data_obs, '.')
    plt.plot(std, '.g')
    plt.show()


    # ---------------------------------------------------------------------------------

    # tiling by frequency

    #

    print("Creating tiles ... ")
    local_misfits = []
    tile_count = 0
    data_ordering = []
    active_cells = actinds
    #np.percentile(np.abs(data_obs), 10, interpolation='lower')
    m0 = (np.ones(mesh.nC) * np.log(sigma_back))[actinds]

    # To run with tiling
    idx_start = 0
    idx_end = 0
    # do every 2 sources
    sources_range = 2
    cnt = 0
    tile_id = 0
    local_misfits = []
    for ii, frequency in enumerate(survey.frequencies):
        if cnt == 0:
            sources_tile = []
            dobs_tile = []
            std_tile = []

        sources = survey.get_sources_by_frequency(frequency)
        sources_tile.extend(sources)

        dobs_tile.extend(data_for_tile[str(frequency)])
        std_tile.extend(std_for_tile[str(frequency)])

        cnt += 1
        if cnt == sources_range:
            delayed_misfit = create_tile_em_misfit(
                sources_tile,
                np.asarray(dobs_tile),
                np.asarray(std_tile),
                mesh, active_cells, tile_id, m0,
                use_global=True
            )
            cnt = 0
            tile_id += 1
            local_misfits += [delayed_misfit]

    # ---------------------------------------------------------------------------------

    # inversion

    #

    # Clean sensitivity function formed with true resistivity
    sim._Jmatrix = None

    # Data Misfit
    coolingFactor = 2
    coolingRate = 1
    beta0_ratio = 1e1

    # Create global data object    
    global_data_object = data.Data(
        survey,
        dobs=data_obs,
        standard_deviation=std,
    )

    global_misfit = objective_function.ComboObjectiveFunction(
                    local_misfits
    )

    # Map for a regularization
    regmap = maps.IdentityMap(nP=int(actinds.sum()))
    reg = regularization.Sparse(mesh, indActive=actinds, mapping=regmap)

    # surface_weights = get_surface_weights(mesh, active_cells, w_fac, octree=False)

    print('[INFO] Getting things started on inversion...')
    # set alpha length scales
    reg.alpha_s = 1
    reg.alpha_x = 1
    reg.alpha_y = 1
    reg.alpha_z = 1

    opt = optimization.ProjectedGNCG(maxIter=10, upper=np.inf, lower=-np.inf)
    invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)
    beta = directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, method="old")
    target = directives.TargetMisfit()
    target.target = survey.nD / 2.
    saveIter = directives.SaveModelEveryIteration()
    saveIterVar = directives.SaveOutputEveryIteration()

    update_Jacobi = directives.UpdatePreconditioner()
    updateSensW = directives.UpdateSensitivityWeights()

    update_IRLS = directives.Update_IRLS(
        f_min_change=1e-4, max_irls_iterations=0, coolEpsFact=1.5, beta_tol=4., coolingRate=coolingRate,
        coolingFactor=coolingFactor
    )

    directiveList = [
         updateSensW, update_IRLS, update_Jacobi, betaest, target  # , saveIter, saveIterVar 
    ]

    inv = inversion.BaseInversion(
        invProb, directiveList=directiveList)
    # opt.LSshorten = 0.5
    opt.remember('xc')

    # -------------------------------------------------------------------------------

    # Run Inversion

    #

    minv = inv.run(m0) 
    print("Total runtime: ", time.time() - tc)
    
    # -------------------------------------------------------------------------------

    # Plot inversion results

    #

    rho_est = actmap * minv
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    mtrue = 1 / np.exp(rho_est)  # np.ones(mesh.nC) * 1 / np.median(rho_app)                        # conductivity
    mtrue[~actinds] = np.nan
    clim = [0, 2000]

    dat = mesh.plotImage((mtrue), ax=ax, grid=False, clim=clim,
                        pcolorOpts={"cmap": "rainbow_r"}
                        )
    ax.set_title('Resistivity')
    cb = plt.colorbar(dat[0], ax=ax)
    # ax.set_xlim([-300, 4300])
    # ax.set_ylim([-400, 0])
    ax.set_aspect('equal')
    ax.plot(
        rx_locs[:, 0],
        rx_locs[:, 1], 'k.'
    )
    plt.title("Sparse Inversion")
    plt.show()



if __name__ == '__main__':

    run()