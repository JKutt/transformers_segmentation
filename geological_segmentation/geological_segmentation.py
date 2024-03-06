from SimPEG import regularization
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
from scipy.ndimage import laplace


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
    for _ in range(num_iterations):
        # Calculate the Laplacian of the input matrix
        laplacian = laplace(input_matrix)

        # Update the input matrix using the Laplacian and a smoothing factor (alpha)
        input_matrix += alpha * laplacian

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
        
        a = 5
        b = 10 #self.segmentations[id]['bbox'][0] - (self.segmentations[id]['bbox'][2] // 2)
        y0 = self.segmentations[id]['bbox'][0]
        x0 = self.segmentations[id]['bbox'][1] - b
        x1 = x0 + self.segmentations[id]['bbox'][3] + (3 * b)
        y1 = y0 + self.segmentations[id]['bbox'][2] + a

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

        grad = minimum_curvature(np.reshape(grad.copy(), self.mesh.shape_cells, order='F'))
        
        return grad.flatten(order='F')

    
#     def update_gradients(self, xc):

#         masks = self.segmentation_model.fit(xc)

#         # loop through masks and assign rotations
#         for ii in range(1, len(masks)):
#             seg_data = masks[ii]['segmentation']
#             seg_data = np.flip(seg_data)
#             # Find the coordinates of the object pixels
#             object_pixels = np.argwhere(seg_data == 1)

#             # Apply PPCA to determine orientation
#             if len(object_pixels) > 1:
#                 # Standardize the data
#                 scaler = StandardScaler()
#                 object_pixels_std = scaler.fit_transform(object_pixels)

#                 # Apply PPCA
#                 pca = PCA(n_components=2)
#                 pca.fit(object_pixels_std)

#                 # The first principal component (eigenvector) will represent the orientation
#                 orientation_vector = pca.components_[0]

#                 # Compute the angle of the orientation vector (in degrees)
#                 angle_degrees = np.arctan2(orientation_vector[1], orientation_vector[0]) * 180 / np.pi

#                 print(f"Orientation angle (degrees): {angle_degrees}")
#                 angle_radians = angle_degrees * np.pi / 180

#                 # Create the 2x2 rotation matrix
#                 # rotation_matrix = np.array([
#                 #     [np.cos(angle_radians), -np.sin(angle_radians)],
#                 #     [np.sin(angle_radians), np.cos(angle_radians)]
#                 # ])
#                 reg_dirs = [np.identity(2) for _ in range(self.mesh.nC)]
#                 sqrt2 = np.sqrt(2)
#                 rotation_matrix = 1 / np.array([[sqrt2, -sqrt2], [-sqrt2, -sqrt2],])
#                 alphas = np.ones((self.mesh.n_cells, self.mesh.dim))
#                 # check for rotation application method
#                 if self.method == 'bound_box':
#                     bbox_mask = self.segmentation_model.get_bound_box_indicies(ii)
#                     for ii in range(self.mesh.nC):

#                         if bbox_mask[ii] == 1:
#                             print('adjusting')
#                             # reg_cell_dirs[ii] = np.array([[cos, -sin], [sin, cos],])
#                             reg_dirs[ii] = rotation_matrix
#                             alphas[ii] = [150, 25]
# #         alphas[ii] = [150, 25]
#                     # reg_dirs[bbox_mask] = [rotation_matrix] * int(bbox_mask.sum())
#                 else:
#                     reg_dirs[seg_data] = [rotation_matrix] * seg_data.sum()

#                 reg_dirs = validate_ndarray_with_shape(
#                     "reg_dirs",
#                     reg_dirs,
#                     shape=[(self.mesh.dim, self.mesh.dim), ("*", self.mesh.dim, self.mesh.dim)],
#                     dtype=float,
#                 )
#                 # now do the alphas
#                 # alphas = np.ones((self.mesh.n_cells, self.mesh.dim))
#                 # alphas[bbox_mask] = [125, 25]
#                 anis_alpha = alphas
#                 mesh = self.mesh
#                 n_active_cells = self.regularization_mesh.n_cells
#                 if reg_dirs.shape == (mesh.dim, mesh.dim):
#                     reg_dirs = np.tile(reg_dirs, (mesh.n_cells, 1, 1))
#                 if reg_dirs.shape[0] != mesh.n_cells:
#                     # check if I need to expand from active cells to all cells (needed for discretize)
#                     if (
#                         reg_dirs.shape[0] == n_active_cells
#                         and self.active_cells is not None
#                     ):
#                         reg_dirs_temp = np.zeros((mesh.n_cells, mesh.dim, mesh.dim))
#                         reg_dirs_temp[self.active_cells] = reg_dirs
#                         reg_dirs = reg_dirs_temp
#                     else:
#                         raise IndexError(
#                             f"`reg_dirs` first dimension, {reg_dirs.shape[0]}, must be either number "
#                             f"of active cells {mesh.n_cells}, or the number of mesh cells {mesh.n_cells}. "
#                         )

#                 # create a stack of matrices of dir @ alphas @ dir.T
#                 anis_alpha = np.einsum("ink,ik,imk->inm", reg_dirs, anis_alpha, reg_dirs)
#                 # Then select the upper diagonal components for input to discretize
#                 if mesh.dim == 2:
#                     anis_alpha = np.stack(
#                         (
#                             anis_alpha[..., 0, 0],
#                             anis_alpha[..., 1, 1],
#                             anis_alpha[..., 0, 1],
#                         ),
#                         axis=-1,
#                     )
#                 elif mesh.dim == 3:
#                     anis_alpha = np.stack(
#                         (
#                             anis_alpha[..., 0, 0],
#                             anis_alpha[..., 1, 1],
#                             anis_alpha[..., 2, 2],
#                             anis_alpha[..., 0, 1],
#                             anis_alpha[..., 0, 2],
#                             anis_alpha[..., 1, 2],
#                         ),
#                         axis=-1,
#                     )
#                 self._anis_alpha = anis_alpha

#             else:
#                 raise ValueError("Not enough object pixels to determine orientation.")

