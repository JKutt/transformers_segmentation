# core python 
import geological_segmentation as geoseg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
from SimPEG.utils.code_utils import validate_ndarray_with_shape

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
            1 / np.exp(self.invProb.reg.objfcts[0].reference_model), ax=ax[1],
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

        # ax[2].hist(1 / np.exp(self.opt.xc), 100)
        ax[2].hist(self.opt.xc, 100)
        # ax[2].set_aspect(1)

        # ax[0].set_ylim([-15,0])
        # ax[0].set_xlim([-15,15])
        ax[0].set_aspect(1)
        # ax[1].set_ylim([-15,0])
        # ax[1].set_xlim([-15,15])
        ax[1].set_aspect(1)
        fig.savefig(f'./geological_segmentation/mcd_iterations/{self.start}.png')
        np.save(f'./geological_segmentation/mcd_iterations/model_{self.start}.npy', self.opt.xc)
        # plt.show()
        self.start += 1


# update the neighbors
class segment_iter(directives.InversionDirective):

    seg_iter = [13]
    segmentation_model: geoseg.SamClassificationModel=None
    method = 'bound_box'
    reg_rots = np.zeros(0)

    def rotation_matrix(self, angle):
        theta = np.radians(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    
    def initialize(self):
        self.count = 0
    
    def endIter(self):

        print(f"Segmenting-iteration: {self.opt.iter}")
        if self.opt.iter in self.seg_iter:
            
            # self.reg[1][1].update_gradients(self.opt.xc)
            masks = self.segmentation_model.fit(self.opt.xc)

            mesh = self.segmentation_model.mesh
            reg_dirs = [np.identity(2) for _ in range(mesh.nC)]
            sqrt2 = np.sqrt(2)
            reg_rots = np.zeros(mesh.nC) + 90

            # loop through masks and assign rotations
            for ii in range(1, len(masks)):
                seg_data = masks[ii]['segmentation']
                seg_data = np.flip(seg_data)
                # Find the coordinates of the object pixels
                object_pixels = np.argwhere(seg_data == 1)

                # Apply PPCA to determine orientation
                if len(object_pixels) > 1:

                    # Apply PPCA
                    pca = PCA(n_components=2)
                    pca.fit(object_pixels)

                    # The first principal component (eigenvector) will represent the orientation
                    orientation_vector = pca.components_[0]
                    scales = pca.singular_values_

                    # Compute the angle of the orientation vector (in degrees)
                    angle_degrees = np.arctan2(orientation_vector[1], orientation_vector[0]) * 180 / np.pi

                    print(f"Orientation angle (degrees): {angle_degrees} and scales: {scales}")
                    angle_radians = angle_degrees * np.pi / 180
                    
                    # rotation_matrix = 1 / np.array([[sqrt2, -sqrt2], [-sqrt2, -sqrt2],])
                    alphas = np.ones((mesh.n_cells, mesh.dim))
                    # check for rotation application method
                    if self.method == 'bound_box':
                        bbox_mask = self.segmentation_model.get_bound_box_indicies(ii)

                        flatten = bbox_mask # masks[ii]['segmentation'].flatten(order='F')
                        reshape = flatten.reshape(mesh.shape_cells, order='F')

                        # plt.imshow(reshape.T)
                        # plt.title(f'mask: {ii + 1}')
                        # plt.gca().invert_yaxis()
                        # # plt.plot([x0, x1], [y0, y1], 'ok')
                        # plt.show()

                        for ii in range(mesh.nC):

                            if bbox_mask[ii] == 1:
                                # print('adjusting')
                                # reg_cell_dirs[ii] = np.array([[cos, -sin], [sin, cos],])
                                reg_rots[ii] = angle_degrees
                                alphas[ii] =  [scales[1], scales[0]* 3]

                        smoothed_rots = geoseg.gaussian_curvature(reg_rots.reshape((len(mesh.h[0]),len(mesh.h[1])), order="F"), smoothness=8).flatten(order="F")
                        # now assign the priciple axis to the rotation matrix
                        for ii in range(mesh.nC):

                            if np.abs(reg_rots[ii]) > 0:
                                # print('adjusting')
                                # reg_cell_dirs[ii] = np.array([[cos, -sin], [sin, cos],])
                                rot_matrix = self.rotation_matrix(smoothed_rots[ii])
                                rotation_axis = np.array([rot_matrix[1, :].tolist(), rot_matrix[0, :].tolist(),])
                                reg_dirs[ii] = rotation_axis
                                
                        #         alphas[ii] = [150, 25]
                        # reg_dirs[bbox_mask] = [rotation_matrix] * int(bbox_mask.sum())
                    else:
                        reg_dirs[seg_data] = [rotation_matrix] * seg_data.sum()

                    reg_dirs = validate_ndarray_with_shape(
                        "reg_dirs",
                        reg_dirs,
                        shape=[(mesh.dim, mesh.dim), ("*", mesh.dim, mesh.dim)],
                        dtype=float,
                    )
                    
                    reg_seg = geoseg.GeologicalSegmentation(

                        mesh, 
                        reg_dirs=reg_dirs,
                        alphas=alphas,
                        ortho_check=False,

                    )

                    reg_small = regularization.Smallness(
                        mesh=mesh,
                        reference_model=self.reg[0][1].reference_model,
                        )


                    self.reg = reg_small + reg_seg
                    self.reg.multipliers = np.r_[1e-5, 1000.0]
                    self.invProb.reg = self.reg
                    # self.invProb.beta = 100.0
                    self.reg_rots = smoothed_rots

                else:
                    raise ValueError("Not enough object pixels to determine orientation.")


        
        self.count += 1



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
ip_dobs = ip_survey.dobs

ip_survey = ip.from_dc_to_ip_survey(ip_survey)
ip_survey.dobs = ip_dobs
ip_survey.dobs = ip_survey.dobs * 0.1/70  #  * dc_survey.dobs

# load mesh
mesh = discretize.TensorMesh.read_UBC("/home/juanito/Documents/projects/mcD/dflt/dcinv2d.msh")
actind = np.ones(mesh.n_cells, dtype=bool)

from SimPEG.electromagnetics.static import utils as sutils
from scipy.spatial import cKDTree

values = np.array([100, 100, 100, 75, 50, 30, 20, 15])

n_layer = values.size

uniqXYlocs, topoCC = sutils.gettopoCC(mesh, actind, option='center')

tree = cKDTree(mesh.gridCC)

# if octree:
#     d, inds = tree.query(np.c_[uniqXYlocs, topoCC])

# else:
d, inds = tree.query(np.c_[uniqXYlocs.gridCC, topoCC])

# Regularization (just for mesh use)
regmap = maps.IdentityMap(nP=int(actind.sum()))

reg = regularization.Sparse(
    mesh, active_cells=actind,
    mapping=regmap
)

surface_weights_temp = np.ones(mesh.nC)
surface_weights_temp[inds] = values[0]
surface_weights = surface_weights_temp.copy()

if n_layer > 1:

    for i in range(n_layer - 1):
        temp = np.zeros(mesh.nC)

        temp[actind] = reg.regularization_mesh.aveFy2CC * reg.regularization_mesh.cell_gradient_y * surface_weights_temp[actind]

        inds = temp == 0.5

        surface_weights[inds] = values[i + 1]

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
standard_deviations_ip = np.abs(ip_survey.dobs) * 0.11 + np.quantile(ip_survey.dobs, 0.65)

dc_data = Data(
    survey=dc_survey, 
    dobs=dc_survey.dobs,
    standard_deviation=standard_deviations
)

ip_data = Data(
    survey=ip_survey, 
    dobs=ip_survey.dobs,
    standard_deviation=standard_deviations_ip
)

dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)
# dmis.w = 1 / np.abs(dc_data.dobs * 0.05 + np.quantile(np.abs(dc_data.dobs), 0.1))
m0 = np.log(1/apparent_resistivity_from_voltage(dc_survey, dc_data.dobs).mean()) * np.ones(mapping.nP)
print(f"background: {np.mean(apparent_resistivity_from_voltage(dc_survey, dc_data.dobs))} ohm-m")
# m0 = np.load(r"/home/juanito/Documents/git/jresearch/geological_segmentation/guided/model_11.npy")
# Create the regularization with GMM information
idenMap = maps.IdentityMap(nP=m0.shape[0])
wires = maps.Wires(('m', m0.shape[0]))

reg_seg = geoseg.GeologicalSegmentation(
    mesh, 
    reg_dirs=None,
    ortho_check=False,
)

reg_small = regularization.Smallness(
    mesh=mesh,
    reference_model=m0,
)

# Weighting
reg_org = regularization.WeightedLeastSquares(
    mesh, 
    active_cells=actind,
    mapping=idenMap,
    reference_model=m0
)

# reg_mean = reg_small + reg_seg # reg_1storder
# reg_mean.multipliers = np.r_[0.00001, 10.0]
reg_mean = reg_org
# # reg_mean.alpha_s = 1e-3
# # reg_mean.alpha_x = 100
# # reg_mean.alpha_y = 100
# # # reg_mean.mrefInSmooth = True
# # reg_mean.approx_gradient = True
# reg_mean.cell_weights = mesh.cell_volumes[actind] * surface_weights[actind]


# Optimization
opt = optimization.ProjectedGNCG(maxIter=25, upper=np.inf, lower=-np.inf, tolCG=1E-5, maxIterLS=20, )
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
updateSensW = directives.UpdateSensitivityWeights(threshold_value=5e-3, every_iteration=False)
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

# mcluster = np.load(r"/home/juanito/Dropbox/JohnLindsey/geoseg/mcd_iterations/model_8.npy")


# fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# mesh.plot_image(1/ np.exp(mcluster), ax=ax, pcolor_opts={'cmap':'Spectral'}, clim=[10, 500])
# ax.axis('equal')
# fig.savefig('mcdermont-dc.png')

# print(dc_survey.source_list)

# ------------------- IP -------------------

# # Model parameters to all cells
# chargeability_map = maps.InjectActiveCells(mesh, actind, 0.0)

# # ip simulation
# simulation_ip = ip.Simulation2DNodal(
    
#     mesh, 
#     survey=ip_survey, 
#     sigmaMap=mapping,
#     solver=Solver,
#     sigma=mapping * mcluster,
#     etaMap=chargeability_map,
#     nky=8

# )

# dmis_ip = data_misfit.L2DataMisfit(data=ip_data, simulation=simulation_ip)
# # dmis.w = 1 / np.abs(dc_data.dobs * 0.05 + np.quantile(np.abs(dc_data.dobs), 0.1))
# m0_ip = 1e-3 * np.ones(actind.sum())
# # m0 = np.load(r"/home/juanito/Documents/git/jresearch/geological_segmentation/guided/model_11.npy")
# # Create the regularization with GMM information
# idenMap_ip = maps.IdentityMap(nP=m0_ip.shape[0])
# wires_ip = maps.Wires(('m', m0_ip.shape[0]))

# # reg_seg_ip = geoseg.GeologicalSegmentation(
# #     mesh, 
# #     reg_dirs=None,
# #     ortho_check=False,
# # )

# # reg_small_ip = regularization.Smallness(
# #     mesh=mesh,
# #     reference_model=m0_ip,
# # )

# # Weighting
# reg_org_ip = regularization.WeightedLeastSquares(
#     mesh, 
#     active_cells=actind,
#     mapping=idenMap_ip,
#     reference_model=m0_ip
# )

# # reg_mean_ip = reg_small_ip + reg_seg_ip # reg_1storder
# # reg_mean_ip.multipliers = np.r_[0.00001, 10.0]
# reg_mean_ip = reg_org_ip
# reg_mean_ip.alpha_s = 1e-3
# reg_mean_ip.alpha_x = 100
# reg_mean_ip.alpha_y = 100
# # # reg_mean.mrefInSmooth = True
# # reg_mean.approx_gradient = True
# # reg_mean.cell_weights = mesh.cell_volumes[actind] * surface_weights[actind]


# # Optimization
# opt_ip = optimization.ProjectedGNCG(maxIter=15, lower=0, upper=0.1, tolCG=1E-5, maxIterLS=20, )
# opt_ip.remember('xc')

# # Set the inverse problem
# invProb_ip = inverse_problem.BaseInvProblem(dmis_ip,  reg_mean_ip,  opt_ip)
# betaIt = directives.BetaSchedule(coolingFactor=2, coolingRate=2)
# targets = directives.MultiTargetMisfits(
#     TriggerSmall=True,
#     TriggerTheta=False,
#     verbose=True,
# )
# MrefInSmooth = directives.PGI_AddMrefInSmooth(verbose=True,  wait_till_stable=True, tolerance=0.0)

# # update_sam = segment_iter()
# # update_sam.segmentation_model = segmentor
# plot_iter_mref = plot_mref()
# plot_iter_mref.mesh = mesh
# updateSensW = directives.UpdateSensitivityWeights(threshold_value=5e-3, every_iteration=False)
# invProb_ip.beta = 1e-1
# inv_ip = inversion.BaseInversion(invProb_ip,
#                             directiveList=[
#                                             updateSensW,
#                                             # update_sam,
#                                             #  petrodir,
#                                             targets, betaIt,
#                                             #  MrefInSmooth,
#                                             plot_iter_mref,
#                                             # update_sam,
#                                             #  save_pgi,
#                                             # update_Jacobi,
#                                             ])

# # Run!

# mcluster_ip = inv_ip.run(m0_ip)
