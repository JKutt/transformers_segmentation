import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile

from discretize import TensorMesh

from SimPEG import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils import plot_1d_layer_model
import multiprocessing
import time
from scipy.sparse import diags
from pymatsolver import Pardiso as Solver

mpl.rcParams.update({"font.size": 16})

def perform_rto(simulation, mesh, perturbed_model, initial_x ):

    std = 0.02 * np.abs(simulation.survey.dobs)

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
        mesh, alpha_s=1e-2, alpha_x=1, reference_model=perturbed_model
    )
    # reg_rto = regularization.Sparse(
    #     mesh, alpha_s=1, alpha_x=1e-2, reference_model=perturbed_model
    # )

    # now determine best beta
    dmis_eigen = utils.eigenvalue_by_power_iteration(dmis_rto, initial_x)

    reg_eigen = utils.eigenvalue_by_power_iteration(reg_rto, initial_x)

    ratio = np.asarray(dmis_eigen / reg_eigen)
    beta = 1e-6 * ratio

    # print(f'beta is: {beta}')

    # Optimization
    opt = optimization.ProjectedGNCG(maxIter=10, upper=np.inf, lower=-np.inf, tolCG=1E-5, maxIterLS=12, )
    opt.remember('xc')

    # Set the inverse problem
    invProb = inverse_problem.BaseInvProblem(dmis_rto,  reg_rto,  opt)
    invProb.startup(initial_x)
    invProb.beta = beta

    return opt.minimize(invProb.evalFunction, initial_x)

def run():

    # storage bucket where we have the data
    data_source = "https://storage.googleapis.com/simpeg/doc-assets/dcr1d.tar.gz"

    # download the data
    downloaded_data = utils.download(data_source, overwrite=True)

    # unzip the tarfile
    tar = tarfile.open(downloaded_data, "r")
    tar.extractall()
    tar.close()

    # path to the directory containing our data
    dir_path = downloaded_data.split(".")[0] + os.path.sep

    # files to work with
    data_filename = dir_path + "app_res_1d_data.dobs"

    # Load data
    dobs = np.loadtxt(str(data_filename))

    A_electrodes = dobs[:, 0:3]
    B_electrodes = dobs[:, 3:6]
    M_electrodes = dobs[:, 6:9]
    N_electrodes = dobs[:, 9:12]
    dobs = dobs[:, -1]

    # Define survey
    unique_tx, k = np.unique(np.c_[A_electrodes, B_electrodes], axis=0, return_index=True)
    n_sources = len(k)
    k = np.sort(k)
    k = np.r_[k, len(k) + 1]

    source_list = []
    for ii in range(0, n_sources):
        # MN electrode locations for receivers. Each is an (N, 3) numpy array
        M_locations = M_electrodes[k[ii] : k[ii + 1], :]
        N_locations = N_electrodes[k[ii] : k[ii + 1], :]
        receiver_list = [
            dc.receivers.Dipole(
                M_locations,
                N_locations,
                data_type="apparent_resistivity",
            )
        ]

        # AB electrode locations for source. Each is a (1, 3) numpy array
        A_location = A_electrodes[k[ii], :]
        B_location = B_electrodes[k[ii], :]
        source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))

    # Define survey
    survey = dc.Survey(source_list)

    # Plot apparent resistivities on sounding curve as a function of Wenner separation
    # parameter.
    electrode_separations = 0.5 * np.sqrt(
        np.sum((survey.locations_a - survey.locations_b) ** 2, axis=1)
    )

    std = 0.02 * np.abs(dobs)

    # data_object = data.Data(survey, dobs=dobs.copy(), standard_deviation=std)

    # Define layer thicknesses
    layer_thicknesses = 5 * np.logspace(0, 1, 25)

    # Define a mesh for plotting and regularization.
    mesh = TensorMesh([(np.r_[layer_thicknesses, layer_thicknesses[-1]])], "0")

    # Define model. A resistivity (Ohm meters) or conductivity (S/m) for each layer.
    starting_model = np.log(2e2 * np.ones((len(layer_thicknesses) + 1)))

    # Define mapping from model to active cells.
    model_map = maps.IdentityMap(nP=len(starting_model)) * maps.ExpMap()
    survey.dobs = dobs.copy()
    simulation = dc.simulation_1d.Simulation1DLayers(
        survey=survey,
        rhoMap=model_map,
        thicknesses=layer_thicknesses,
    )

    # -------------------------------------------------------------------------------------------------

    # rto

    #

    beta_perturb = 1e4
    # Wm = np.sqrt(beta_perturb) * np.eye(mesh.nC)

    n_model_samples = mesh.nC

    # coefficient matrix
    zero_means_ = np.zeros(1)
    identity_matrix_ = np.eye(1)

    Wm = np.sqrt(beta_perturb) * diags(np.ones(n_model_samples))

    # print('creating s')
    s = np.random.multivariate_normal(zero_means_, identity_matrix_, size=n_model_samples)

    # perturbed_mod = np.linalg.solve(Wm, s3)
    ainv = Solver(Wm)

    perturbed_mod = ainv * s

    results = [None] * 500
    draws_beta = [None] * 500
    TKO = False

    process_pool = multiprocessing.Pool(8)
    rto_tasks = [None] * 500

    print('start processes')

    for ii in range(500):

        rto_tasks[ii] = process_pool.apply_async(
                        perform_rto,
                        (
                            simulation,
                            mesh,
                            perturbed_mod,
                            starting_model,
                        )
                    )

    print('finished launch')
    process_pool.close()
    process_pool.join()  
    print('getting rto models')
    for ii in range(500):

        results[ii] = rto_tasks[ii].get()

    # recovered_model = np.vstack(results).mean(axis=0)
    np.save('rto_models.npy', np.vstack(results))


if __name__ == '__main__':
    start = time.time()
    run()
    print(f'completed in: {time.time() - start} seconds')
