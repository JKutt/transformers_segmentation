from SimPEG import regularization, utils
from SimPEG.utils.code_utils import validate_ndarray_with_shape
import discretize
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator
from PIL import Image
from scipy import stats
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
from scipy.ndimage import laplace, gaussian_filter


def gaussian_curvature(matrix, smoothness=1):
    """
        Apply Gaussian curvature smoothing to a 2D matrix.

        Parameters:
            matrix (array-like): The input 2D matrix.
            smoothness (float, optional): The amount of blurring to apply (default is 1).
            
        Returns:
            array-like: The smoothed matrix after Gaussian curvature smoothing.

        Notes:
            This function applies Gaussian curvature smoothing to the input matrix. 
            Gaussian curvature smoothing helps in reducing noise and enhancing features 
            in the matrix by convolving it with a Gaussian kernel.

        Example:
            >>> matrix = np.array([[1, 2, 3],
            ...                    [4, 5, 6],
            ...                    [7, 8, 9]])
            >>> smoothed_matrix = gaussian_curvature(matrix, smoothness=2)
    """
    # Perform minimum curvature interpolation
    interpolated_matrix = np.copy(matrix)

    # Apply blurring to the interpolated matrix
    if smoothness > 0:
        interpolated_matrix = gaussian_filter(interpolated_matrix, sigma=smoothness)

    return interpolated_matrix


def latin_hypercube_subsampling(samples, n_subsamples):
    """
    Perform Latin Hypercube subsampling on the given set of samples.

    Parameters:
    - samples (numpy.ndarray): Input samples of shape (n, num_dimensions).
    - n_subsamples (int): Number of subsamples to generate.

    Returns:
    - numpy.ndarray: Subsamples selected using Latin Hypercube Sampling, with shape (n_subsamples, num_dimensions).
    """
    n = samples.shape[0]

    # Generate random indices for each dimension
    random_indices = np.random.permutation(n)

    # Sort the indices along each dimension
    sorted_indices = np.argsort(random_indices)

    # Take the first n_subsamples indices along each dimension
    subsample_indices = sorted_indices[:n_subsamples]

    # Use the selected indices to extract the subsamples
    subsamples = samples[subsample_indices]

    return subsamples


def minimum_curvature(
        input_matrix:np.ndarray, 
        num_iterations:int=100, 
        alpha:float=0.1
) -> np.ndarray:
    """

        Perform a minimum curvature operation on the given input matrix.

        Parameters:
        - input_matrix (numpy.ndarray): The input matrix on which the minimum curvature operation is performed.
        - num_iterations (int, optional): The number of iterations for the curvature smoothing process (default is 100).
        - alpha (float, optional): The smoothing factor controlling the influence of the Laplacian (default is 0.1).

        Returns:
        numpy.ndarray: The input matrix after applying the minimum curvature operation.

    """
    input_matrix = gaussian_curvature(input_matrix, smoothness=1)
    # for _ in range(num_iterations):
    #     # Calculate the Laplacian of the input matrix
    #     laplacian = laplace(input_matrix)

    #     # Update the input matrix using the Laplacian and a smoothing factor (alpha)
    #     input_matrix += alpha * laplacian

    return input_matrix


def calculate_iou(
        mask1:np.ndarray,
        mask2:np.ndarray
) -> float:
    """
    Calculate the Intersection over Union (IoU) between two binary masks.

    Parameters:
        mask1 (numpy.ndarray): The first binary mask.
        mask2 (numpy.ndarray): The second binary mask.

    Returns:
        float: The Intersection over Union (IoU) score.
    """
    
    # Ensure the masks have the same shape
    if mask1.shape != mask2.shape:
        raise ValueError("Mask shapes do not match.")

    # Convert masks to binary (0 or 1) values
    mask1 = np.array(mask1 > 0, dtype=np.uint8)
    mask2 = np.array(mask2 > 0, dtype=np.uint8)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    iou = intersection / union if union > 0 else 0.0
    return iou


# interjecting the sam classification
class SamClassificationModel():
    """

        A class representing a SamClassificationModel for segmentation and prediction tasks.

        Parameters:
        - mesh (mesh object): The mesh object for the model.
        - kneighbors (int, optional): The number of neighbors for the model (default is 20).
        - segmentation_model_checkpoint (str, optional): The path to the segmentation model checkpoint (default is
        "/home/juanito/Documents/trained_models/sam_vit_h_4b8939.pth").
        - proportions_factor (float, optional): The factor controlling the influence of the Laplacian in the segmentation
        process (default is 1e-5).

        Attributes:
        - segmentation_model_checkpoint (str): The path to the segmentation model checkpoint.
        - mesh: The mesh object for the model.
        - kneighbors (int): The number of neighbors for the model.
        - indexpoint (numpy.ndarray): A matrix to hold information about overlapping masks.
        - portions_factor (float): The factor controlling the influence of the Laplacian in the segmentation process.
        - mask_assignment: A matrix to assign masks to each cell.
        - mask_generator: An instance of the SamAutomaticMaskGenerator.

        Methods:
        - fit(model: np.ndarray) -> dict: Fits the model using the provided data and returns the results.
        - predict(model: np.ndarray) -> np.ndarray: Predicts outcomes based on the provided model.
        - query_neighbours(cell_number: int) -> np.ndarray: Queries the neighbors of a specific cell.
        - update_weights_matrix(new_matrix: np.ndarray) -> None: Updates the weights matrix of the model.

    """

    def __init__(
        self,
        mesh,
        segmentation_model_checkpoint: str=r"/home/juanito/Documents/trained_models/sam_vit_h_4b8939.pth",
        proportions_factor: float=1e-5,
    ):
        
        self.segmentation_model_checkpoint = segmentation_model_checkpoint
        self.mesh = mesh
        self.portions_factor = proportions_factor
        self.mask_assignment = None
        self.segmentations = None

        # load segmentation network model 
        sam = sam_model_registry["vit_h"](checkpoint=self.segmentation_model_checkpoint)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # sam.to(device=device)
        self.segment_model = sam

    def fit(
            self, 
            model:np.ndarray=None,
    ) -> list[dict]:
        """

            Fit the model using the provided data.

            Parameters:
            - model (numpy.ndarray): The input model data.

            Returns:
            list: list of dictionaries for each segment.

    """
        if model is None:

            raise ValueError('need a model')
        
        else:

            model_normalized = np.exp(model) / np.abs(np.exp(model)).max()

            image_rgb = Image.fromarray(np.uint8(cm.jet(model_normalized.reshape(self.mesh.shape_cells, order='F'))*255))
            image_rgb = image_rgb.convert('RGB')

            # get the segments
            mask_generator = SamAutomaticMaskGenerator(self.segment_model)
            results = mask_generator.generate(np.asarray(image_rgb))

            # ---------------------------------------------------------------------------------------------

            # create a matrix that holds information about overlapping mask if they happen to

            # this is done using intersection over union method

            #

            nlayers = len(results)

            union_matrix = np.zeros((nlayers, nlayers))
            votes = np.zeros(nlayers)
            for ii in range(nlayers):
                for jj in range(nlayers):
                    iou_score = calculate_iou(
                        results[ii]['segmentation'],
                        results[jj]['segmentation']
                    )
                    union_matrix[ii, jj] = iou_score

            # # check mask filter option
            # if True:
            #     # find the background and isolated objects
            #     background_index = np.where((union_matrix[:, 0] == 1) | (union_matrix[:, 0] == 0))

            #     filtered_masks = [ results[seg] for seg in background_index[0]]
            #     filtered_union = union_matrix[background_index[0], :]
            #     filtered_union = filtered_union[:, background_index[0]]
            #     results = filtered_masks
            #     union_matrix = filtered_union
            #     nlayers = len(results)

            # OK lets use the the union matrix as weights but deal with nested masks
            portions_factor = self.portions_factor
            for ii in range(nlayers):
                mask_tally = 0
                find_1 = False
                for jj in range(nlayers - 1):

                    if union_matrix[ii, jj] == 1:

                        if union_matrix[ii, jj+1] < union_matrix[ii, jj] and union_matrix[ii, jj+1] != 0:

                            mask_tally += 1
                        else:

                            union_matrix[ii, :jj] = union_matrix[ii, :jj] * portions_factor

                        break

                if union_matrix[ii, jj + 1] == 1:

                    union_matrix[ii, :jj+1] = union_matrix[ii, :jj+1] * portions_factor

                print(f'mask {ii} vote total: {mask_tally}')
                votes[ii] = mask_tally

            # ----------------------------------------------------------------------

            # now that we have the votes lets contstruct the global weights matrix

            # and assign the mask to each cell

            #

            # find where we have votes
            indices = np.where(votes == 1)

            diagonals_modify = indices[0][1:]
            print(diagonals_modify)

            weight_matrix = union_matrix.copy()

            weight_matrix[diagonals_modify, diagonals_modify] = 0.0

            print(weight_matrix)

        self.segmentations = results
        self.weights_matrix = weight_matrix
        self.union_matrix = union_matrix

        return results
    
    def predict(

            self, 
            model:np.ndarray,
            # gmm:utils.WeightedGaussianMixture

    ) -> np.ndarray:
        """

            Predict outcomes based on the provided model.

            Parameters:
            - model (numpy.ndarray): The input model data.

            Returns:
            numpy.ndarray: Predicted outcomes defing the quasi-geophysical model.

        """

        # --------------------------------------------------------------------------------------

        # assign each cell a mask to assign it's neighbors

        #

        outcomes = np.arange(len(self.segmentations))

        nlayers = len(self.segmentations)

        # now lets get the values for each cell
        physical_property = np.zeros(nlayers)
        for ii in range(nlayers):

            idx = np.vstack(np.where(self.segmentations[ii]['segmentation'].flatten(order='F') == True))[0]

            value = model[idx].mean()

            physical_property[ii] = value

            print(f'mask {ii}: {1/np.exp(value)} ohm - m')

        hx, hy = self.mesh.shape_cells
        x = np.arange(hx)
        y = np.arange(hy)
        xx, yy = np.meshgrid(x, y)

        mask_locations = np.vstack([xx.flatten(), yy.flatten()])

        mask_assignment = np.zeros(mask_locations.shape[1], dtype=int)
        quasi_geological_model = np.ones(mask_locations.shape[1]) * physical_property[0]

        for ii in range(mask_locations.shape[1]):

            for jj in range(1, nlayers):

                idx = np.vstack(np.where(self.segmentations[jj]['segmentation'] == True))

                point_set = idx.T

                distances = np.sqrt(np.sum((point_set - mask_locations[:, ii].T)**2, axis=1))

                min_distance = np.min(distances)
                
                if min_distance == 0:
                    mask_assignment[ii] = random.choices(outcomes, self.weights_matrix[jj, :], k=1)[0]
                    quasi_geological_model[ii] = physical_property[mask_assignment[ii]]

        # plt.hist(mask_assignment)

        self.mask_assignment = mask_assignment

        return quasi_geological_model
   
    def get_focused_mask(
            self,
            bound_box_id: int,
            
    ) -> tuple:
        
        mask_predictor = SamPredictor(self.segment_model)
        
        return (mask_predictor.predict(
            box=self.segmentations[bound_box_id]['bbox'],
            multimask_output=True
        ))
    
    def get_bound_box_indicies(

            self,
            id: int,

    ) -> np.ndarray:
        """
            Returns a flattened array representing the bounding box indices of the specified segmentation.

            Parameters:
            - id (int): The identifier of the segmentation within the 'segmentations' attribute.

            Returns:
            np.ndarray: A flattened array where the indices corresponding to the bounding box of the
                    specified segmentation are set to 1, and the rest are set to 0. The array is
                    flattened in Fortran order ('F').
        """
        
        a = 0
        b = 0 #self.segmentations[id]['bbox'][0] - (self.segmentations[id]['bbox'][2] // 2)
        y0 = self.segmentations[id]['bbox'][0]
        x0 = self.segmentations[id]['bbox'][1] - (2 * b)
        x1 = x0 + self.segmentations[id]['bbox'][3] + (4 * b)
        y1 = y0 + self.segmentations[id]['bbox'][2] + a

        # check that the numbers checkout
        if y0 < 0:
            y0 = 0
        if y1 > self.mesh.shape_cells[1]:
            y1 = self.mesh.shape_cells[1] - 1
        if x0 < 0:
            x0 = 0
        if x1 > self.mesh.shape_cells[0]:
            x1 = self.mesh.shape_cells[0] - 1

        # generate as sparse matrix when things get big
        bbox = np.zeros(self.segmentations[id]['segmentation'].shape)

        bbox[x0:x1, y0:y1] = 1

        return bbox.flatten(order='F')
    
    def query_neighbours(
            
            self, 
            cell_number: int=0
    
    ) -> np.ndarray:
        """

            Query the neighbors of a specific cell.

            Parameters:
            - cell_number (int): The cell number.

            Returns:
            numpy.ndarray: Locations of neighbors.

        """

        # check that cell number is in bounds
        try:
            
            return np.vstack(
                np.where(
                    self.segmentations[
                        self.mask_assignment[cell_number]
                    ]['segmentation'] == True
                )
            )
        
        except ValueError as e:

            raise ValueError(f'cell number is not within bounds: {e}')
    
    def update_weights_matrix(
            self,
            new_matrix: np.ndarray,
    ) -> None:
        """

            Update the weights matrix of the model.

            Parameters:
            - new_matrix (numpy.ndarray): The new weights matrix.

        """
        
        self.weights_matrix = new_matrix


class GeologicalSegmentation(regularization.SmoothnessFullGradient):

    def __init__(
        self, 
        mesh: discretize.TensorMesh, 
        alphas: np.ndarray=None, 
        reg_dirs: np.ndarray=None, 
        ortho_check: bool=True,
        method: str='bound_box',
        **kwargs
    ):
        super().__init__(
            mesh=mesh, 
            alphas=alphas, 
            reg_dirs=reg_dirs, 
            ortho_check=ortho_check, 
            **kwargs)

        self.mesh = mesh
        self.method = method
    
    def deriv(self, m):
        m_d = self.mapping.deriv(self._delta_m(m))
        G = self.cell_gradient
        M_f = self.W
        r = G @ (self.mapping * (self._delta_m(m)))

        grad = m_d.T * (G.T @ (M_f @ r))

        # grad = minimum_curvature(np.reshape(grad.copy(), self.mesh.shape_cells, order='F'))
        
        return grad.flatten(order='F')

    
class GaussianMixtureMarkovRandomField(utils.WeightedGaussianMixture):

    def __init__(
        self,
        n_components,
        mesh,
        actv=None,
        kdtree=None,
        indexneighbors=None,
        boreholeidx=None,
        T=12.,
        masks=None,
        kneighbors=0,
        norm=2,
        init_params='kmeans',
        max_iter=100,
        covariance_type='full',
        means_init=None,
        n_init=10, 
        precisions_init=None,
        random_state=None, 
        reg_covar=1e-06, 
        tol=0.001, 
        verbose=0,
        verbose_interval=10, 
        warm_start=False, 
        weights_init=None,
        anisotropy=None,
        index_anisotropy=None, # Dictionary with anisotropy and index
        index_kdtree=None,# List of KDtree
        segmentation_model_checkpoint=r"C:\Users\johnk\Documents\git\jresearch\PGI\dcip\sam_vit_h_4b8939.pth",
        #**kwargs
    ):

        super(GaussianMixtureMarkovRandomField, self).__init__(
            n_components=n_components,
            mesh=mesh,
            actv=actv,
            covariance_type=covariance_type,
            init_params=init_params,
            max_iter=max_iter,
            means_init=means_init,
            n_init=n_init,
            precisions_init=precisions_init,
            random_state=random_state,
            reg_covar=reg_covar,
            tol=tol,
            verbose=verbose,
            verbose_interval=verbose_interval,
            warm_start=warm_start,
            weights_init=weights_init,
            #boreholeidx=boreholeidx
            # **kwargs
        )
        # setKwargs(self, **kwargs)
        self.kneighbors = kneighbors
        self.T = T
        self.boreholeidx = boreholeidx
        self.anisotropy = anisotropy
        self.norm = norm
        self.masks = masks

        # load segmentation network model
        sam = sam_model_registry["vit_h"](checkpoint=segmentation_model_checkpoint)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

        if self.mesh.gridCC.ndim == 1:
            xyz = np.c_[self.mesh.gridCC]
        elif self.anisotropy is not None:
            xyz = self.anisotropy.dot(self.mesh.gridCC.T).T
        else:
            xyz = self.mesh.gridCC
        
        if self.actv is None:
            self.xyz = xyz
        else:
            self.xyz = xyz[self.actv]
        
        if kdtree is None:
            print('Computing KDTree, it may take several minutes.')
            self.kdtree = spatial.KDTree(self.xyz)
        else:
            self.kdtree = kdtree
        
        if indexneighbors is None:
            print('Computing neighbors, it may take several minutes.')
            _, self.indexneighbors = self.kdtree.query(self.xyz, k=self.kneighbors+1, p=self.norm)
        else:
            self.indexneighbors = indexneighbors

        self.indexpoint = copy.deepcopy(self.indexneighbors)
        self.index_anisotropy = index_anisotropy
        self.index_kdtree = index_kdtree
        if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:

            self.unitxyz = []
            for i, anis in enumerate(self.index_anisotropy['anisotropy']):
                self.unitxyz.append((anis).dot(self.xyz.T).T)

            if self.index_kdtree is None:
                self.index_kdtree = []
                print('Computing rock unit specific KDTree, it may take several minutes.')
                for i, anis in enumerate(self.index_anisotropy['anisotropy']):
                    self.index_kdtree.append(spatial.KDTree(self.unitxyz[i]))

            #print('Computing new neighbors based on rock units, it may take several minutes.')
            #for i, unitindex in enumerate(self.index_anisotropy['index']):
        #        _, self.indexpoint[unitindex] = self.index_kdtree[i].query(self.unitxyz[i][unitindex], k=self.kneighbors+1)

    def update_neighbors_index(
            
            self,
            model:np.ndarray
            
    ) -> None:
        """
        
            method that segments the input model and assigns new neighbors described
            by the segmentation map

            :param model: geophysical model
            :type model: np.ndarray

        """

        model_normalized = np.exp(model) / np.abs(np.exp(model)).max()

        image_rgb = Image.fromarray(np.uint8(cm.jet(model_normalized.reshape(self.mesh.shape_cells, order='F'))*255))
        image_rgb = image_rgb.convert('RGB')

        result = self.mask_generator.generate(np.asarray(image_rgb))


        # ---------------------------------------------------------------------------------------------

        # create a matrix that holds information about overlapping mask if they happen to

        # this is done using intersection over union method

        #

        nlayers = self.kneighbors

        union_matrix = np.zeros((nlayers, nlayers))
        for ii in range(nlayers):
            for jj in range(nlayers):
                iou_score = calculate_iou(result[ii]['segmentation'], result[jj]['segmentation'])
                union_matrix[ii, jj] = iou_score
                print("IoU score:", iou_score)

        # ------------------------------------------------------------------------------------

        # modify the overlap matrix to assign the proper neighbors mask in the case of onions

        #

        sub_union_matrix = union_matrix[1:, 1:].copy()

        # calculate how many non zero in a row of our overlap matrix
        for jj in range(sub_onion.shape[0]):

            if np.count_nonzero(sub_onion[jj, :]) > 1:

                mask_index = np.nonzero(sub_union_matrix[jj, :])
                print(mask_index[0][-1])
                sub_union_matrix[jj, mask_index[0][-1]] = 1
                sub_union_matrix[jj, mask_index[0][0]] = 0

        # --------------------------------------------------------------------------------------

        # assign each cell a mask to assign it's neighbors

        #

        hx, hy = mesh.shape_cells
        x = np.arange(hx)
        y = np.arange(hy)
        xx, yy = np.meshgrid(x, y)

        mask_locations = np.vstack([xx.flatten(), yy.flatten()])

        mask_assignment = np.zeros(mask_locations.shape[1])

        for ii in range(mask_locations.shape[1]):

            for jj in range(nlayers - 1):

                idx = np.vstack(np.where(result[jj + 1]['segmentation'] == True))

                point_set = idx.T

                # print(point_set.shape, np.vstack(idx).shape, xx.shape)
                distances = np.sqrt(np.sum((point_set - mask_locations[:, ii].T)**2, axis=1))
                # print(jj, mask_assignment[:, ii].T, point_set[0, :])
                min_distance = np.min(distances)
                
                if min_distance == 0:
                    mask_assignment[ii] = jj + 1

        # ----------------------------------------------------------------------------------------

        # now update the indexpoint matrix

        #

        for kk in range(mask_assignment.shape[0]):

            # check union matrix for the correct mask
            union_index = mask_assignment[kk] - 1
            mask_select = np.nonzero(sub_union_matrix[union_index, :])[0][0]

            idx = np.vstack(np.where(result[mask_select]['segmentation'].flatten(order='F') == True))[0]
            shape_idx = idx.shape[0]

            # if the mask is smaller than the user defined number of neighbors
            if idx.shape[0] < (self.kneighbors + 1):

                self.indexpoint[kk, :] = self.indexpoint[kk, 0]
                self.indexpoint[kk, -shape_idx:] = idx

            # otherwise assign the entire mask
            else:

                self.indexpoint[kk, :] = idx[:(self.kneighbors + 1)]


    def computeG(self, z, w, X):

        #Find neighbors given the current state of data and model
        if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:
            prediction = self.predict(X)
            unit_index = []
            for i in range(self.n_components):
                unit_index.append(np.where(prediction==i)[0])
            for i, unitindex in enumerate(unit_index):
                _, self.indexpoint[unitindex] = self.index_kdtree[i].query(
                    self.unitxyz[i][unitindex],
                    k=self.kneighbors+1,
                    p=self.index_anisotropy['norm'][i]
                )

        logG = (self.T/(2.*(self.kneighbors+1))) * (
            (z[self.indexpoint] + w[self.indexpoint]).sum(
                axis=1
            )
        )
        return logG

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
        _, self.means_, self.covariances_ = (
            self._estimate_gaussian_parameters(X, self.mesh, np.exp(log_resp), self.reg_covar,self.covariance_type)
        )
        #self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

        logweights = logsumexp(np.c_[[log_resp, self.computeG(np.exp(log_resp), self.weights_,X)]], axis=0)
        logweights = logweights - logsumexp(
            logweights, axis=1, keepdims=True
        )

        self.weights_ = np.exp(logweights)
        if self.boreholeidx is not None:
            aux = np.zeros((self.boreholeidx.shape[0],self.n_components))
            aux[np.arange(len(aux)), self.boreholeidx[:,1]]=1
            self.weights_[self.boreholeidx[:,0]] = aux


    def _check_weights(self, weights, n_components, n_samples):
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
        weights = check_array(
            weights, dtype=[np.float64, np.float32],
            ensure_2d=True
        )
        _check_shape(weights, (n_components, n_samples), 'weights')

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        n_samples, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = self._check_weights(
                self.weights_init,
                n_samples,
                self.n_components
            )

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
            X, self.mesh, resp, self.reg_covar, self.covariance_type)
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
