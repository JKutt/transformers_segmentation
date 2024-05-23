# core python 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ipywidgets
import os

# tools in the simPEG Ecosystem 
import discretize  # for creating computational meshes

# linear solvers
try: 
    from pymatsolver import Pardiso as Solver  # this is a fast linear solver 
except ImportError:
    from SimPEG import SolverLU as Solver  # this will be slower

# SimPEG inversion machinery
from SimPEG import (
    Data, maps,
    data_misfit, regularization, optimization, inverse_problem, 
    inversion, directives
) 

# DC resistivity and IP modules
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import induced_polarization as ip

from SimPEG.electromagnetics.static.utils.static_utils import (
    apparent_resistivity_from_voltage,
    plot_pseudosection,
)

# set the font size in the plots
from matplotlib import rcParams
rcParams["font.size"] = 14


class plot_mref(directives.InversionDirective):

    mesh = None
    
    def initialize(self):
        self.start = 0
        # self.endIter()
    
    def endIter(self):
        # plot
        meshCore = self.mesh
        # predicted = self.invProb.reg.gmmref.predict(self.opt.xc.reshape(-1, 1))
        fig,ax = plt.subplots(3,1,figsize=(15,5))
        mm = meshCore.plot_image(
            self.opt.xc, ax=ax[0],
            # clim=[-np.log(250),-np.log(10),],
            # clim=[0,500],
            pcolor_opts={'cmap':'Spectral_r'}
        )
        # fig,ax = plt.subplots(1,1,figsize=(15,5))
        mm2 = meshCore.plot_image(
            1 / np.exp(self.invProb.reg.objfcts[0].mref), ax=ax[1],
            # clim=[-np.log(250),-np.log(10),],
            clim=[0,500],
            pcolor_opts={'cmap':'Spectral_r'}
        )
        # ax.set_xlim([-750,750])
        # ax.set_ylim([-250,0])
        # fig,ax = plt.subplots(1,1,figsize=(15,5))
        # mmpred = meshCore.plot_image(
        #    predicted, ax=ax[3],
        #     # clim=[-np.log(250),-np.log(10),],
        #     pcolor_opts={'cmap':'Spectral'}
        # )
        
        #plt.colorbar(mm[0])
        # utils.plot2Ddata(
        #     meshCore.gridCC,1 / np.exp(mtrue[actcore]),nx=500,ny=500,
        #     contourOpts={'alpha':0},
        #     clim=[0,500],
        #     ax=ax[0],
        #     level=True,
        #     ncontour=2,
        #     levelOpts={'colors':'k','linewidths':2,'linestyles':'--'},
        #     method='nearest'
        # )

        ax[2].hist(1 / np.exp(self.opt.xc), 100)
        # ax[2].set_aspect(1)

        # ax[0].set_ylim([-15,0])
        # ax[0].set_xlim([-15,15])
        ax[0].set_aspect(1)
        # ax[1].set_ylim([-15,0])
        # ax[1].set_xlim([-15,15])
        ax[1].set_aspect(1)
        fig.savefig(f'./geological_segmentation/mcd_iterations/{self.start}.png')
        np.save(f'./geological_segmentation/mcd_iterations/model_{self.start}.npy', self.opt.xc)
        plt.show()
        self.start += 1


def read_dcip_data(filename, verbose=True):
    """
    Read in a .OBS file from the Century data set into a python dictionary. 
    The format is the old UBC-GIF DCIP format.
    
    Parameters
    ----------
    filename : str
        Path to the file to be parsed
    
    verbose: bool
        Print some things? 
    
    
    Returns
    -------
    dict
        A dictionary with the locations of
        - a_locations: the positive source electrode locations (numpy array) 
        - b_locations: the negative source electrode locations (numpy array) 
        - m_locations: the receiver locations (list of numpy arrays)
        - n_locations: the receiver locations (list of numpy arrays)
        - observed_data: observed data (list of numpy arrays)
        - standard_deviations: assigned standard deviations (list of numpy arrays)
        - n_sources: number of sources (int)
    
    """
    
    # read in the text file as a numpy array of strings (each row is an entry)
    contents = np.genfromtxt(filename, skip_header=2)

    unique_rows = np.unique(contents[:, :2], axis=0)

    dc_data = {}

    for i, row in enumerate(unique_rows):

        print(f"Source {i}: A-loc: {row[0]}, B-loc: {row[1]}")

        dc_data[f"{row[0]} {row[1]}"] = {'a': row[0], 'b': row[1], 'm': [], 'n': [], 'd': []}

    for ii in range(contents.shape[0]):

        dc_data[f"{contents[ii, 0]} {contents[ii, 1]}"]['m'] += [contents[ii, 2]]
        dc_data[f"{contents[ii, 0]} {contents[ii, 1]}"]['n'] += [contents[ii, 3]]
        dc_data[f"{contents[ii, 0]} {contents[ii, 1]}"]['d'] += [contents[ii, 4]]

    source_list = []

    sources = list(dc_data.keys())
    
    for jj in range(len(dc_data.keys())):
    
        # receiver electrode locations in 2D 
        m_locs = np.vstack([
            dc_data[sources[jj]]['m'], 
            np.zeros_like(dc_data[sources[jj]]["m"])
        ]).T
        n_locs = np.vstack([
            dc_data[sources[jj]]["n"],
            np.zeros_like(dc_data[sources[jj]]["n"])
        ]).T
        
        # construct the receiver object 
        receivers = dc.receivers.Dipole(locations_m=m_locs, locations_n=n_locs)
        
        # construct the source 
        source = dc.sources.Dipole(
            location_a=np.r_[dc_data[sources[jj]]["a"], 0.],
            location_b=np.r_[dc_data[sources[jj]]["b"], 0.],
            receiver_list=[receivers]
        )
        
        # append the new source to the source list
        source_list.append(source)

        survey_obj = dc.Survey(source_list=source_list)
        survey_obj.dobs = contents[:, 4]

    return survey_obj


dc_data_file = "/home/juanito/Documents/projects/mcD/mcd-dc.dat"
ip_data_file = "/home/juanito/Documents/projects/mcD/mcd-ip.dat"

dc_survey = read_dcip_data(dc_data_file)
ip_survey = read_dcip_data(ip_data_file)

# load mesh
mesh = discretize.TensorMesh.read_UBC("/home/juanito/Documents/projects/mcD/dflt/dcinv2d.msh")
actind = np.ones(mesh.n_cells, dtype=bool)

# set up simulation
# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = maps.ExpMap(mesh)
mapactive = maps.InjectActiveCells(
    
    mesh=mesh,
    indActive=actind,
    valInactive=np.log(1e-8)

)
mapping = expmap * mapactive
simulation = dc.Simulation2DNodal(
    
    mesh, 
    survey=dc_survey, 
    sigmaMap=mapping,
    solver=Solver,
    nky=8

)

standard_deviations = np.abs(dc_survey.dobs) * 0.08 + np.quantile(dc_survey.dobs, 0.1)
# standard_deviations = np.abs(ip_survey.dobs) * 0.08 + np.quantile(ip_survey.dobs, 0.1)

dc_data = Data(
    survey=dc_survey, 
    dobs=dc_survey.dobs,
    standard_deviation=standard_deviations
)

# Plot voltages pseudo-section
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    dc_survey,
    dobs=apparent_resistivity_from_voltage(dc_survey, dc_data.dobs),
    plot_type="scatter",
    ax=ax1,
    scale="log",
    cbar_label="V/A",
    scatter_opts={"cmap": 'Spectral_r'},
)
ax1.set_title("Normalized Voltages")
plt.show()

plt.hist(np.log(dc_data.dobs), bins=50)
plt.axvline(np.log(np.quantile(dc_data.dobs, 0.1)), color='r')
plt.show()

dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)
# dmis.w = 1 / np.abs(dc_data.dobs * 0.05 + np.quantile(np.abs(dc_data.dobs), 0.1))
m0 = np.log(1/apparent_resistivity_from_voltage(dc_survey, dc_data.dobs).mean()) * np.ones(mapping.nP)
# m0 = np.load(r"/home/juanito/Documents/git/jresearch/geological_segmentation/guided/model_11.npy")
# Create the regularization with GMM information
idenMap = maps.IdentityMap(nP=m0.shape[0])
wires = maps.Wires(('m', m0.shape[0]))

# # load mask
# rot_mask = np.load('rotation_block_mask_scaled.npy')

# # set the regularization
# alphas = np.ones((meshCore.n_cells, meshCore.dim))
# # alphas[rot_mask] = [125, 25]
# alphas[meshCore.cell_centers[:, 1] < 0.5] = [125, 25]
# sqrt2 = np.sqrt(2)
# # reg_cell_dirs = 1 / np.array([[sqrt2, -sqrt2], [sqrt2, sqrt2],])
# # lets just assign them to the dip structure
# # reg_cell_dirs = [np.identity(2) for _ in range(meshCore.nC)]

# # lets just assign them to the dip structure
# reg_cell_dirs = [1 / np.array([[sqrt2, sqrt2], [sqrt2, -sqrt2],]) for _ in range(meshCore.nC)]
# print(reg_cell_dirs)
# # lets expand the area we want to
# # Dike 45*
# dike00 = mesh.gridCC[:,1] > fault_function(mesh.gridCC[:,0],1, 50)
# dike01 = mesh.gridCC[:,1] < fault_function(mesh.gridCC[:,0],1, 255)
# dike_dir_reg = np.logical_and(dike00,dike01)

# # reg model
# reg_model = model.copy()

# reg_model[dike_dir_reg]=4

# # cos = np.cos(140*np.pi / 180) * 2
# # sin = np.sin(140*np.pi / 180) * 2

# for ii in range(meshCore.nC):

#     if rot_mask[ii] == 1:
#         print('adjusting')
#         # reg_cell_dirs[ii] = np.array([[cos, -sin], [sin, cos],])
#         reg_cell_dirs[ii] = 1 / np.array([[sqrt2, -sqrt2], [-sqrt2, -sqrt2],])
#         alphas[ii] = [150, 25]

# reg_seg = geoseg.GeologicalSegmentation(
#     meshCore, 
#     reg_dirs=None,
#     ortho_check=False,
# )

# reg_1storder = regularization.SmoothnessFullGradient(
#     meshCore, 
#     reg_dirs=reg_cell_dirs,
#     alphas=alphas,
#     ortho_check=False,
#     reference_model=np.log(1/dcutils.apparent_resistivity_from_voltage(survey, dc_data.dobs).mean()) * np.ones(mapping.nP)
# )

# reg_small = regularization.Smallness(
#     mesh=meshCore,
#     reference_model=np.log(1/dcutils.apparent_resistivity_from_voltage(survey, dc_data.dobs).mean()) * np.ones(mapping.nP),
# )

# # Weighting
reg_org = regularization.WeightedLeastSquares(
    mesh, 
    active_cells=actind,
    mapping=idenMap,
    reference_model=m0
)

# reg_mean = reg_small + reg_seg # reg_1storder
# reg_mean.multipliers = np.r_[0.00001, 10.0]
reg_mean = reg_org
reg_mean.alpha_s = 1e-3
reg_mean.alpha_x = 100
reg_mean.alpha_y = 100
# # reg_mean.mrefInSmooth = True
# reg_mean.approx_gradient = True


# Optimization
opt = optimization.ProjectedGNCG(maxIter=15, upper=np.inf, lower=-np.inf, tolCG=1E-5, maxIterLS=20, )
opt.remember('xc')

# Set the inverse problem
invProb = inverse_problem.BaseInvProblem(dmis,  reg_mean,  opt)
betaIt = directives.BetaSchedule(coolingFactor=2, coolingRate=4)
targets = directives.MultiTargetMisfits(
    TriggerSmall=True,
    TriggerTheta=False,
    verbose=True,
)
MrefInSmooth = directives.PGI_AddMrefInSmooth(verbose=True,  wait_till_stable=True, tolerance=0.0)

# update_sam = segment_iter()
# update_sam.segmentation_model = segmentor
plot_iter_mref = plot_mref()
plot_iter_mref.mesh = mesh
# updateSensW = directives.UpdateSensitivityWeights(threshold=5e-1, everyIter=False)
# update_Jacobi = directives.UpdatePreconditioner()
# save_pgi = SavePGIOutput('./pgi_param')
invProb.beta = 1e-1
inv = inversion.BaseInversion(invProb,
                            directiveList=[
                                            # updateSensW,
                                            # update_sam,
                                            #  petrodir,
                                            targets, betaIt,
                                            #  MrefInSmooth,
                                            plot_iter_mref,
                                            # update_sam,
                                            #  save_pgi,
                                            # update_Jacobi,
                                            ])

# Run!

mcluster = inv.run(m0)


fig, ax = plt.subplots(1, 1, figsize=(10, 5))

mesh.plot_image(1/ np.exp(mcluster), ax=ax, pcolor_opts={'cmap':'Spectral'}, clim=[10, 500])
ax.axis('equal')
fig.savefig('mcdermont-dc.png')

print(dc_survey.source_list)
