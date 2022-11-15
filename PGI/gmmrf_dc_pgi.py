import numpy as np
import matplotlib.pyplot as plt
from SimPEG import (
    maps,
    data,
    utils,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
)
from SimPEG.electromagnetics.static import resistivity as dc, utils as dcutils
import numpy as np
import copy
import matplotlib.pyplot as plt
from pymatsolver import PardisoSolver
from scipy.stats import norm

# from PGI_DC_example_Utils import plot_pseudoSection, getCylinderPoints
from skimage import data, color
from skimage.transform import  rescale, resize, downscale_local_mean
import scipy.ndimage as ndi
import discretize
from sklearn.mixture import GaussianMixture
from scipy import spatial, linalg
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky, _compute_log_det_cholesky,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_tied,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_spherical,
    _check_means,
    _check_precisions,
    _check_shape,
)
from sklearn.utils import check_array
from scipy.special import logsumexp


# -----------------------------------------------------------------------

# creating directives and additional classes

#

#plot learned mref
class Plot_mref(directives.InversionDirective):
    
    def initialize(self):
        self.endIter()
    
    def endIter(self):
        # plot
        fig,ax = plt.subplots(1,1,figsize=(15,5))
        mm = meshCore.plotImage(
            self.invProb.reg.objfcts[0].mref, ax=ax,
            clim=[-np.log(250),-np.log(10),],
            pcolorOpts={'cmap':'viridis'}
        )
        ax.set_xlim([-750,750])
        ax.set_ylim([-250,0])
        ax.set_aspect(1)
        #plt.colorbar(mm[0])
        utils.plot2Ddata(
            meshCore.gridCC,mtrue[actcore],nx=500,ny=500,
             contourOpts={'alpha':0},
             #clim=[0,5],
             ax=ax,
             level=True,
             ncontour=2,
             levelOpts={'colors':'k','linewidths':2,'linestyles':'--'},
            method='nearest'
        )
        plt.show()

class GaussianMixtureMarkovRandomField(GaussianMixture):

    def __init__(
        self, 
        n_components,
        mesh, # the mesh is used to measure the distance between points and find neighboring pixels
        beta = 12.,
        kneighbors=0,
        covariance_type='full',
        init_params='kmeans', max_iter=100,
        means_init=None, n_init=10, precisions_init=None,
        random_state=None, reg_covar=1e-06, tol=0.001, verbose=0,
        verbose_interval=10, warm_start=False, weights_init=None,
        #**kwargs
    ):
        self.mesh = mesh
        self.kneighbors = kneighbors
        print('Computing KDTree, it may take several minutes.')
        self.tree = spatial.KDTree(self.mesh.gridCC)
        _, self.indexpoint = self.tree.query(self.mesh.gridCC, k=self.kneighbors+1)#, distance_upper_bound=100.)
        self.beta = beta
        
        super(GaussianMixtureMarkovRandomField, self).__init__(
            covariance_type=covariance_type,
            init_params=init_params,
            max_iter=max_iter,
            means_init=means_init,
            n_components=n_components,
            n_init=n_init,
            precisions_init=precisions_init,
            random_state=random_state,
            reg_covar=reg_covar,
            tol=tol,
            verbose=verbose,
            verbose_interval=verbose_interval,
            warm_start=warm_start,
            weights_init=weights_init,
            #**kwargs
        )
        # setKwargs(self, **kwargs)
        
    def computeG(self,z,w):
            logG = (self.beta/(2.*(self.kneighbors+1))) * ((z[self.indexpoint] + w[self.indexpoint]).sum(axis=1))
            
            return logG
        
    def computeB(self, A):
        beta2 = np.zeros(A.shape[1])
        for k in range(A.shape[1]):
            beta2[k] = ((A[:,k][self.indexpoint[:,1:]]- Utils.mkvc(A[:,k],numDims=2))**2.).sum()/self.mesh.nC
        print(beta2)
        return beta2

    def computeA(self, A, z, beta):
        perm = np.random.permutation(self.mesh.nC)
        for i in perm:
            for j in range(self.n_components):
                notjindx = np.where(np.linspace(0,j,j,endpoint=False,dtype='int') != j)
                Aij = A[i,notjindx].sum()
                amj = A[self.indexpoint[i,1:],j].sum()
                coeff2 = Aij - ( amj / self.kneighbors )
                coeff1 = Aij * amj / self.kneighbors
                coeff0 = - 0.5 * z[i,j] * Aij * beta[j] / self.kneighbors
                roots = np.roots(np.r_[1., coeff2, coeff1, coeff0])
                roots = roots[np.isreal(roots)]
                roots = np.real(roots)
                A[i,j] = np.r_[0.,np.real(roots)].max()
                    
        return A
                    
    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        _ , self.means_, self.covariances_ = (
            self._estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,self.covariance_type)
        )
        #self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)
        
        logweights = logsumexp(np.c_[[log_resp, self.computeG(np.exp(log_resp),self.weights_)]],axis=0)
        logweights = logweights - logsumexp(logweights,axis=1,keepdims=True)
        self.weights_= np.exp(logweights)
           
    def _estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type):
        """Estimate the Gaussian distribution parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data array.
        resp : array-like, shape (n_samples, n_components)
            The responsibilities for each data sample in X.
        reg_covar : float
            The regularization added to the diagonal of the covariance matrices.
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.
        Returns
        -------
        nk : array-like, shape (n_components,)
            The numbers of data samples in the current components.
        means : array-like, shape (n_components, n_features)
            The centers of the current components.
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        """
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = {"full": _estimate_gaussian_covariances_full,
                   "tied": _estimate_gaussian_covariances_tied,
                   "diag": _estimate_gaussian_covariances_diag,
                   "spherical": _estimate_gaussian_covariances_spherical
                   }[covariance_type](resp, X, nk, means, reg_covar)
        return resp, means, covariances
    
    def _check_weights(self,weights, n_components,n_samples):
        """Check the user provided 'weights'.
        Parameters
        ----------
        weights : array-like, shape (n_components,)
            The proportions of components of each mixture.
        n_components : int
            Number of components.
        Returns
        -------
        weights : array, shape (n_components,)
        """
        weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=True)
        _check_shape(weights, (n_components,n_samples), 'weights')
        
    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        n_samples, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = self._check_weights(self.weights_init,
                                               n_samples,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)
            
    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights*np.ones((n_samples,self.n_components)) if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init



# -----------------------------------------------------------------------

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

# divide domain by  45* fault at 100 m
fault_function = lambda x, slope, shift: slope * x + shift
domain = np.ones(mesh.nC, dtype='int64')
domain0 = mesh.gridCC[:,1] < fault_function(mesh.gridCC[:,0],-1,100.)
domain[domain0] = 0

# Layered Earth
layered_model0 = 3 * np.ones(mesh.nC, dtype='int64')
layered_model0[mesh.gridCC[:,1]>-50] = 2
layered_model0[mesh.gridCC[:,1]>-25] = 1

layered_model1 = 3 * np.ones(mesh.nC, dtype='int64')
layered_model1[mesh.gridCC[:,1]>-75] = 2
layered_model1[mesh.gridCC[:,1]>-50] = 1

model = layered_model1
model[domain0] = layered_model0[domain0]

# Dike 45*
dike0 = mesh.gridCC[:,1] > fault_function(mesh.gridCC[:,0],1, 100)
dike1 = mesh.gridCC[:,1] < fault_function(mesh.gridCC[:,0],1, 125)
dike = np.logical_and(dike0,dike1)

model[dike]=4

# plot
fig,ax = plt.subplots(1,1,figsize=(10,5))
mm = mesh.plotImage(model, ax=ax, pcolorOpts={'cmap':'viridis'})

plt.gca().set_xlim([-1000,1000])
plt.gca().set_ylim([-250,0])
plt.gca().set_aspect(2)
plt.colorbar(mm[0])

plt.show()

# define conductivities
res_true = np.ones(mesh.nC)
res_true[model==1]= 50
res_true[model==2]= 250
res_true[model==3]= 100
res_true[model==4]= 10

cond_true = 1./res_true

mtrue = np.log(cond_true)

xmin, xmax = -400., 400.
ymin, ymax = -300., 0.
zmin, zmax = 0, 0
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
actcore,  meshCore = utils.mesh_utils.ExtractCoreMesh(xyzlim, mesh)
actind = np.ones_like(actcore)
print(meshCore)

# plot
fig,ax = plt.subplots(1,1,figsize=(10,5))
mm = meshCore.plotImage(
    
    np.log10(cond_true)[actcore],
    ax=ax,
    pcolorOpts={'cmap':'viridis'}

)

utils.plot2Ddata(

    meshCore.gridCC,mtrue[actcore],nx=500,ny=500,
    contourOpts={'alpha':0},
    #clim=[0,5],
    ax=ax,
    level=True,
    ncontour=2,
    levelOpts={'colors':'k','linewidths':2,'linestyles':'--'},
    method='nearest'
    
)
#plt.gca().set_ylim([-200,0])
plt.gca().set_aspect(1)
plt.colorbar(mm[0])
plt.show()

# -----------------------------------------------------------------------

# create a survey for a dc simulation

#

np.linspace(25,250,10)

xmin, xmax = -350., 350.
ymin, ymax = 0., 0.
zmin, zmax = 0, 0

endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
srclist = []

for dipole in np.linspace(25,250,10):
    
    survey1 = dcutils.gen_DCIPsurvey(
        
        endl, survey_type="dipole-dipole",
        dim=mesh.dim,
        a=dipole,
        b=dipole,
        n=16,
        #d2flag="2.5D"
    
    )
    
    srclist +=(survey1.source_list)

survey = dc.Survey(srclist)

# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = maps.ExpMap(mesh)
mapactive = maps.InjectActiveCells(
    
    mesh=mesh,
    indActive=actcore,
    valInactive=-np.log(100)

)
mapping = expmap * mapactive
simulation = dc.simulation_2d.Simulation2DNodal(
    
    mesh, 
    survey=survey, 
    sigmaMap=mapping,
    solver=PardisoSolver,
    nky=8

)

# -----------------------------------------------------------------------

# create synthetic data and view psuedo-section

#

relative_measurement_error = 0.05
dc_data = simulation.make_synthetic_data(
    
    mtrue[actcore],
    relative_error=relative_measurement_error,
    noise_floor=1e-4,
    force=True,
    add_noise=True,

)

relative_error_list = (np.abs(dc_data.standard_deviation/dc_data.dobs))
print(relative_error_list.min())
print(relative_error_list.max())
print(np.median(relative_error_list))
print(relative_error_list.mean())
plt.hist(np.log10(relative_error_list), 50)
plt.show()

# # Plot the pseudosection
# fig, ax = plt.subplots(1,1,figsize=(10,5))
# ax, _= dcutils.plot_pseudosection(
#     dc_data,ax=ax,
#     data_type="appResistivity",
#     scale='log',
#     data_locations=True,
#     sameratio=False, 
#     pcolor_opts={'cmap':'viridis_r'}
# )
# ax.set_title("Pseudosection",fontsize=20)
# plt.show()

# Plot the histogram of the data
fig, ax = plt.subplots(1,1,figsize=(10,5))
hist, edges = np.histogram(-np.log(((DCutils.apparent_resistivity(dc_data)))),bins=50, density=False)
ax.bar(1./np.exp(edges[:-1]), hist, width=np.diff(1./np.exp(edges)), ec="k", align="edge",color='#8172B3');
ax.grid(True,which='both')
ax.grid(True,which="both",ls="-")

ax.set_xlabel('Data: Apparent Resistivity (Ohm-m)',fontsize=16)
ax.tick_params(labelsize=14)
ax.set_ylabel('Occurences',fontsize=16)
ax.set_xticks(np.r_[10,20,50,75,100,125,150])

plt.show()


# -----------------------------------------------------------------------

# carry out PGI inversion with GMMRF

#

# Set the initial model to the true background mean
m0 = -np.log(100.0) * np.ones(mapping.nP)
mref = m0
#m0 = m_pgi_nguyen
#mref = reg_nguyen.objfcts[0].mref

# Create data misfit object
dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)

# Create the regularization with GMM information
idenMap = maps.IdentityMap(nP=m0.shape[0])
wires = maps.Wires(("m", m0.shape[0]))
## By default the PGI regularization uses the least-squares approximation. 
## It requires then the directives.GaussianMixtureUpdateModel() 
reg_mean_potts = regularization.SimplePGI(
    gmmref=clf_nguyen,#reg_nguyen.objfcts[0].gmm, 
    mesh=mesh, wiresmap=wires, maplist=[idenMap], mref=mref, indActive=actcore
)

# Regularization Weighting
alpha_s = 1.
alpha_x = 1.0
alpha_y = 1.0
reg_mean_potts.alpha_s = alpha_s
reg_mean_potts.alpha_x = alpha_x
reg_mean_potts.alpha_y = alpha_y
reg_mean_potts.mrefInSmooth = False

# Optimization
opt = optimization.ProjectedGNCG(
    maxIter=50, lower=-10, upper=10, maxIterLS=20, maxIterCG=20, tolCG=1e-5
)
opt.remember("xc")

# Set the inverse problem
invProb = inverse_problem.BaseInvProblem(dmis, reg_mean_potts, opt)

# Inversion directives
## Beta Strategy with Beta and Alpha
beta_value = directives.BetaEstimate_ByEig(beta0_ratio=10.,n_pw_iter=10)
alphas_value = directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio= np.r_[0,5.,5.,0.5,0.5],verbose=True,n_pw_iter=10,
)
beta_alpha_iteration = directives.PGI_BetaAlphaSchedule(
    verbose=True,
    coolingFactor=2.0,
    tolerance=0.2,  # Tolerance on Phi_d for beta-cooling
    progress=0.2,  # Minimum progress, else beta-cooling
)
## PGI multi-target misfits
targets = directives.MultiTargetMisfits(verbose=True,)
## Put learned reference model in Smoothness once stable
MrefInSmooth = directives.PGI_AddMrefInSmooth(
    verbose=True,
    tolerance=0.001,
)
## PGI update to the GMM, Smallness reference model and weights: 
## **This one is required when using the Least-Squares approximation of PGI 
petrodir = directives.PGI_UpdateParameters(update_gmm=True,zeta=0.)
## Sensitivity weights based on the starting half-space
updateSensW = directives.UpdateSensitivityWeights(threshold=1e-2, everyIter=True)
## Preconditioner
update_Jacobi = directives.UpdatePreconditioner()

# Rule of thumb for Potts model: indiag log coeff = 1/reg_covar of clf
#Rule unit 1 and 2 should be at least 4 cells apart
Pottmatrix = np.zeros([4,4])
#Pottmatrix[0,1] = -1*float(1./clf_nguyen.covariances_[0])**3.
#Pottmatrix[1,0] = -1*float(1./clf_nguyen.covariances_[0])**3.
#Pottmatrix[0,-2] = -1*float(1./clf.covariances_)**3.
#Pottmatrix[-2,0] = -1*float(1./clf.covariances_)**3.
for i in range(Pottmatrix.shape[0]):
    Pottmatrix[i,i] = 1e0
Pottmatrix[-1,-1] = 1e0
#anisotropy = np.eye(2)
#anisotropy[0][0] = 10
#print(anisotropy)

smoothref = directives.PGI_GMMRF_IsingModel(
    neighbors=18, verbose=True,
    max_probanoise=1e-3,
    maxit_factor=5.,
    # maxit=2*mesh.nC,
    Pottmatrix=Pottmatrix,
    anisotropies = {'anisotropy':[anis_vertical,anisotropy,anisotropy,anis_dike],'norm':[2,2,2,2]}#clf_nguyen.index_anisotropy
)

plot_mref = Plot_mref()

# Inversion
inv = inversion.BaseInversion(
    invProb,
    # directives list: the order matters!
    directiveList=[
        updateSensW,
        alphas_value,
        beta_value,
        petrodir,
        targets,
        beta_alpha_iteration,
        smoothref,
        MrefInSmooth,
        update_Jacobi,
        plot_mref,
    ],
)
np.random.seed(518936)
# Run the inversion
m_gmmrf = inv.run(m0)


