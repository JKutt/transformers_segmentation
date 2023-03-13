import torch
import numpy as np


# -----------------------------------------------------------------------------------------------------------------

# utilities

#

def kronecker_product_sparse(
        
        A: torch.sparse_coo_tensor,
        B: torch.sparse_coo_tensor,
        
) -> torch.sparse_coo_tensor:
    """

        Method to compute the Kronecker product of torch sparse matrices
        Based on scipy's sparse Kronecker product

        Parameters
        ----------
        A : sparse matrix of the product
        B : sparse matrix of the product
        format : tensor.sparse_coo_tensor
        
        Returns
        -------
        kronecker product in a tensor sparse coo matrix format
       
    """

    # calculate the output dimensions
    output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

    # determine row and columns and extract the data from sparse matrix A
    row = A.indices()[0, :].repeat_interleave(B.values().shape[0])
    col = A.indices()[1, :].repeat_interleave(B.values().shape[0])
    data = A.values().repeat_interleave(B.values().shape[0])

    # take into account sparse matrix B
    row *= B.shape[0]
    col *= B.shape[1]

    # increment block indices
    row,col = row.reshape(-1,B.values().shape[0]), col.reshape(-1,B.values().shape[0])
    row += B.indices()[0, :]
    col += B.indices()[1, :]
    row,col = row.reshape(-1), col.reshape(-1)

    # compute block entries
    data = data.reshape(-1, B.values().shape[0]) * B.values()
    data = data.reshape(-1)
    
    # return output_shape
    return torch.sparse_coo_tensor(torch.vstack([row, col]), data, output_shape).coalesce()


def av_extrap(n):

        # create end values
        ends = torch.sparse_coo_tensor( ([0, n], [0, n - 1]), [0.5, 0.5], (n + 1, n)).coalesce()

        Av = torch.sparse.spdiags((0.5 * torch.ones(n, 1) * torch.from_numpy(np.asarray([1, 1]))).T,
                                torch.tensor([-1, 0]), (n + 1, n)).coalesce() + ends

        return Av


def ddx_cell_grad(n):

        diag_d = (torch.ones([n, 1]) * torch.from_numpy(np.asarray([-1, 1])))
        
        # diag_d[0, 0] = 0
        # diag_d[-1, -1] = 0
        
        D_ = torch.sparse.spdiags(diag_d.T,
                                torch.tensor([-1, 0]),
                                (n + 1, n)).coalesce()
        
        stencil_values = D_.values()
        stencil_indices = D_.indices()
        stencil_values[0] = 0
        stencil_values[-1] = 0
        D = torch.sparse_coo_tensor(torch.vstack([stencil_indices[0, :], stencil_indices[1, :]]), stencil_values, (n+1, n)).coalesce()

        return D

# -----------------------------------------------------------------------------------------------------------------

# mesh object

#

class torch_mesh:
    """

        mesh object with discretized operators written with pytorch

    """

    def __init__(self, hx: torch.tensor, hy: torch.tensor, hz: torch.tensor):
        """
s
            initialize mesh

        """
        
        n1 = hx.shape[0]
        n2 = hy.shape[0]
        n3 = hz.shape[0]

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.h1 = hx
        self.h2 = hy
        self.h3 = hz
        
        # define single parameters
        self.number_of_cells = n1 * n2 * n3

    def shape_nodes(self):

        return tuple(x + 1 for x in self.shape_cells())
    

    def shape_cells(self):

        return (self.n1, self.n2, self.n3)
    

    def shape_faces_x(self):

        return self.shape_nodes()[:1] + self.shape_cells()[1:]


    def shape_faces_y(self):

        sc = self.shape_cells()
        sn = self.shape_nodes()
        
        return (sc[0], sn[1]) + sc[2:]

    
    def shape_faces_z(self):

        return self.shape_cells()[:2] + self.shape_nodes()[2:]


    def average_cell_to_face(self):

        n1, n2, n3 = self.n1, self.n2, self.n3

        n1_I = torch.sparse.spdiags(torch.ones([n1, 1]).T, torch.tensor([0]), (n1, n1)).coalesce()
        n2_I = torch.sparse.spdiags(torch.ones([n2, 1]).T, torch.tensor([0]), (n2, n2)).coalesce()
        n3_I = torch.sparse.spdiags(torch.ones([n3, 1]).T, torch.tensor([0]), (n3, n3)).coalesce()

        acf = torch.vstack([
                kronecker_product_sparse(
                    kronecker_product_sparse(n3_I, n2_I), av_extrap(n1)),
                kronecker_product_sparse(
                    kronecker_product_sparse(n3_I,
                    av_extrap(n2)), n1_I),
                kronecker_product_sparse(
                    kronecker_product_sparse(av_extrap(n3), n2_I), n1_I)
            ]
        )
        
        return acf


    def cell_gradient_x(self):

        n1, n2, n3 = self.n1, self.n2, self.n3

        # create identity matrices for each dir.
        n2_I = torch.sparse.spdiags(torch.ones([n2, 1]).T, torch.tensor([0]), (n2, n2)).coalesce()
        n3_I = torch.sparse.spdiags(torch.ones([n3, 1]).T, torch.tensor([0]), (n3, n3)).coalesce()

        G1 = kronecker_product_sparse(
                n3_I,
                kronecker_product_sparse(n2_I, ddx_cell_grad(n1))
            )
        
        # Compute areas of cell faces & volumes
        V = self.average_cell_to_face() @ self.cell_volumes()

        # extract the x faces
        face_areas = self.cell_faces().values()
        fa_v = face_areas / V
        nFx = torch.prod(torch.tensor(self.shape_faces_x()))
        nFy = torch.prod(torch.tensor(self.shape_faces_y()))
        nFz = torch.prod(torch.tensor(self.shape_faces_z()))
        nn = (nFx, nFy, nFz)[: 3]
        nn = np.r_[0, nn]
        dim = 0
        start = np.sum(nn[: dim + 1])
        end = np.sum(nn[: dim + 2])

        L = fa_v[start:end]
        
        return torch.sparse.spdiags(L, torch.tensor([0]), (L.shape[0], L.shape[0])) @ G1
    

    def cell_gradient_y(self):

        n1, n2, n3 = self.n1, self.n2, self.n3

        # create identity matrices for each dir.
        n1_I = torch.sparse.spdiags(torch.ones([n1, 1]).T, torch.tensor([0]), (n1, n1)).coalesce()
        n3_I = torch.sparse.spdiags(torch.ones([n3, 1]).T, torch.tensor([0]), (n3, n3)).coalesce()

        G2 = kronecker_product_sparse(
                n3_I,
                kronecker_product_sparse(ddx_cell_grad(n2), n1_I)
            )
        
        # Compute areas of cell faces & volumes
        V = self.average_cell_to_face() @ self.cell_volumes()

        # extract the x faces
        face_areas = self.cell_faces().values()
        fa_v = face_areas / V
        nFx = torch.prod(torch.tensor(self.shape_faces_x()))
        nFy = torch.prod(torch.tensor(self.shape_faces_y()))
        nFz = torch.prod(torch.tensor(self.shape_faces_z()))
        nn = (nFx, nFy, nFz)[: 3]
        nn = np.r_[0, nn]
        dim = 1
        start = np.sum(nn[: dim + 1])
        end = np.sum(nn[: dim + 2])

        L = fa_v[start:end]

        return G2  # torch.sparse.spdiags(L, torch.tensor([0]), (L.shape[0], L.shape[0])) @ G2


    def cell_gradient_z(self):

        n1, n2, n3 = self.n1, self.n2, self.n3

        # create identity matrices for each dir.
        n1_I = torch.sparse.spdiags(torch.ones([n1, 1]).T, torch.tensor([0]), (n1, n1)).coalesce()
        n2_I = torch.sparse.spdiags(torch.ones([n2, 1]).T, torch.tensor([0]), (n2, n2)).coalesce()

        G3 = kronecker_product_sparse(
                ddx_cell_grad(n3),
                kronecker_product_sparse(n2_I, n1_I)
            )

        # Compute areas of cell faces & volumes
        V = self.average_cell_to_face() @ self.cell_volumes()

        # extract the x faces
        face_areas = self.cell_faces().values()
        fa_v = face_areas / V
        nFx = torch.prod(torch.tensor(self.shape_faces_x()))
        nFy = torch.prod(torch.tensor(self.shape_faces_y()))
        nFz = torch.prod(torch.tensor(self.shape_faces_z()))
        nn = (nFx, nFy, nFz)[: 3]
        nn = np.r_[0, nn]
        dim = 2
        start = np.sum(nn[: dim + 1])
        end = np.sum(nn[: dim + 2])

        L = fa_v[start:end]

        return torch.sparse.spdiags(L, torch.tensor([0]), (L.shape[0], L.shape[0])) @ G3


    def nodal_gradient_matrix(self) -> torch.sparse_coo_tensor:
        """

            calculates the gradient matrix

            Returns
            -------
            tensor sparse coo matrix containing the nodal gradient

        """

        n1,n2,n3 = self.n1, self.n2, self.n3

        def ddx(n):
            return torch.sparse.spdiags((torch.ones([n+1,1]) * torch.from_numpy(np.asarray([-1,1]))).T,
                                        torch.tensor([0, 1]),
                                        (n, n+1)).coalesce()

        G1 = kronecker_product_sparse(torch.sparse.spdiags(torch.ones([n3+1, 1]).T, torch.tensor([0]), (n3+1, n3+1)).coalesce(),
                                      kronecker_product_sparse(torch.sparse.spdiags(torch.ones([n2+1, 1]).T,
                                                                                    torch.tensor([0]),
                                                                                    (n2+1, n2+1)).coalesce(),
                                                               ddx(n1)))
        G2 = kronecker_product_sparse(torch.sparse.spdiags(torch.ones([n3+1, 1]).T,
                                                           torch.tensor([0]),
                                                           (n3+1, n3+1)).coalesce(),
                                      kronecker_product_sparse(ddx(n2), torch.sparse.spdiags(torch.ones([n1+1, 1]).T,
                                                                                             torch.tensor([0]), 
                                                                                             (n1+1, n1+1)).coalesce()))
        
        G3 = kronecker_product_sparse(ddx(n3), kronecker_product_sparse(torch.sparse.spdiags(torch.ones([n2+1, 1]).T,
                                                                                             torch.tensor([0]),
                                                                                             (n2+1, n2+1)).coalesce(),
                                                                        torch.sparse.spdiags(torch.ones([n1+1, 1]).T,
                                                                                             torch.tensor([0]),
                                                                                             (n1+1, n1+1)).coalesce()))
        
        # grad on the nodes
        return torch.vstack([G1, G2, G3])
    

    def cell_faces(self) -> torch.sparse_coo_tensor:
        """

            calculates the cell faces

            Returns
            -------
            tensor sparse coo matrix containing the cell volumes

        """

        n1, n2, n3 = self.n1, self.n2, self.n3
        h1, h2, h3 = self.h1, self.h2, self.h3

        # create identity matrices for each dir.
        n1_I = torch.sparse.spdiags(torch.ones([n1 + 1, 1]).T, torch.tensor([0]), (n1 + 1, n1 + 1)).coalesce()
        n2_I = torch.sparse.spdiags(torch.ones([n2 + 1, 1]).T, torch.tensor([0]), (n2 + 1, n2 + 1)).coalesce()
        n3_I = torch.sparse.spdiags(torch.ones([n3 + 1, 1]).T, torch.tensor([0]), (n3 + 1, n3 + 1)).coalesce()
        h1_d = torch.sparse.spdiags(h1, torch.tensor([0]), (n1, n1)).coalesce()
        h2_d = torch.sparse.spdiags(h2, torch.tensor([0]), (n2, n2)).coalesce()
        h3_d = torch.sparse.spdiags(h3, torch.tensor([0]), (n3, n3)).coalesce()

        # do x
        Fx = kronecker_product_sparse(h3_d, kronecker_product_sparse(h2_d, n1_I)).values()

        # do y
        Fy = kronecker_product_sparse(h3_d, kronecker_product_sparse(n2_I, h1_d)).values()

        # do z
        Fz = kronecker_product_sparse(n3_I, kronecker_product_sparse(h2_d, h1_d)).values()

        dims = Fx.shape[0] + Fy.shape[0] + Fz.shape[0]

        return torch.sparse.spdiags(torch.hstack([Fx, Fy, Fz]), torch.tensor([0]), (dims, dims)).coalesce()


    def cell_volumes(self) -> torch.sparse_coo_tensor:
        """

            calculates the cell volumes

            Returns
            -------
            tensor sparse coo matrix containing the cell volumes

        """

        h1, h2, h3 = self.h1, self.h2, self.h3
        
        diag_h1 = torch.sparse.spdiags(h1, torch.tensor([0]), (h1.shape[0], h1.shape[0])).coalesce()
        diag_h2 = torch.sparse.spdiags(h2, torch.tensor([0]), (h2.shape[0], h2.shape[0])).coalesce()
        diag_h3 = torch.sparse.spdiags(h3, torch.tensor([0]), (h3.shape[0], h3.shape[0])).coalesce()
        
        return kronecker_product_sparse(diag_h3, kronecker_product_sparse(diag_h2, diag_h1)).values()
    

    def cell_lengths(self):

        n1, n2, n3 = self.n1, self.n2, self.n3
        h1, h2, h3 = self.h1, self.h2, self.h3

        # create identity matrices for each dir.
        n1_I = torch.sparse.spdiags(torch.ones([n1 + 1, 1]).T, torch.tensor([0]), (n1 + 1, n1 + 1))
        n2_I = torch.sparse.spdiags(torch.ones([n2 + 1, 1]).T, torch.tensor([0]), (n2 + 1, n2 + 1))
        n3_I = torch.sparse.spdiags(torch.ones([n3 + 1, 1]).T, torch.tensor([0]), (n3 + 1, n3 + 1))
        h1_I = torch.sparse.spdiags(h1, torch.tensor([0]), (n1, n1))
        h2_I = torch.sparse.spdiags(h2, torch.tensor([0]), (n2, n2))
        h3_I = torch.sparse.spdiags(h3, torch.tensor([0]), (n3, n3))

        # do x
        Lx = kronecker_product_sparse(n3_I, kronecker_product_sparse(n2_I, h1_I))

        L = sp.diags(sp.hstack([
        sp.kron(sp.eye(n3 + 1), sp.kron(sp.eye(n2 + 1), sp.diags(h1))).diagonal(),
        sp.kron(sp.eye(n3 + 1), sp.kron(sp.diags(h2), sp.eye(n1 + 1))).diagonal(),
        sp.kron(sp.diags(h3), sp.kron(sp.eye(n2 + 1), sp.eye(n1 + 1))).diagonal()]).toarray(), [0]
        )

        return L


    def getFaceToCellCenterMatrix(self) -> torch.sparse_coo_tensor:
        """

            calculates averaging matrix that goes from face to cell centers

            Returns
            -------
            tensor sparse coo matrix containing the face to cell center operators

        """

        n1,n2,n3 = self.n1, self.n2, self.n3

        def av(n):
            return torch.sparse.spdiags((torch.ones(n + 1, 1) * (torch.from_numpy(np.asarray([1, 1])) * 0.5)).T,
                                         torch.tensor([0, 1]), (n, n+1)).coalesce()
        
        # create the sparse identity matrices for each cell direction
        n1_I = torch.sparse.spdiags(torch.ones([n1, 1]).T, torch.tensor([0]), (n1, n1)).coalesce()
        n2_I = torch.sparse.spdiags(torch.ones([n2, 1]).T, torch.tensor([0]), (n2, n2)).coalesce()
        n3_I = torch.sparse.spdiags(torch.ones([n3, 1]).T, torch.tensor([0]), (n3, n3)).coalesce()

        A1 = kronecker_product_sparse(n3_I, kronecker_product_sparse(n2_I,av(n1)))
        A2 = kronecker_product_sparse(n3_I, kronecker_product_sparse(av(n2), n1_I))
        A3 = kronecker_product_sparse(av(n3), kronecker_product_sparse(n2_I, n1_I))
        
        # average from faces to cell-centers
        return torch.hstack([A1, A2, A3])
