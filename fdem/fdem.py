import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class jmesh:
    """

        mesh object with discretized operators

    """

    def __init__(self, n1, n2, n3, h1, h2, h3):
        """

            initialize mesh

        """

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        
        n1 = n1 + 1
        n2 = n2 + 1
        n3 = n3 + 1
        
        # define single parameters
        self.number_of_cells = n1 * n2 * n3
        
        # define the nodal grid
        x_nodal = np.arange(n1) * h1
        y_nodal = np.arange(n2) * h2
        z_nodal = np.arange(n3) * h3

        X_nodal,Y_nodal,Z_nodal = np.meshgrid(x_nodal[0,:], y_nodal[0,:],z_nodal[0,:])
        
        self.vector_of_node_grid_x = X_nodal.flatten()
        self.vector_of_node_grid_y = Y_nodal.flatten()
        self.vector_of_node_grid_z = Z_nodal.flatten()
        
        # define cell centered grid
        x_center = (np.arange(n1-1) + 0.5) * h1
        y_center = (np.arange(n2-1) + 0.5) * h2
        z_center = (np.arange(n3-1) + 0.5) * h3

        X_center,Y_center,Z_center = np.meshgrid(x_center[0,:], y_center[0,:],z_center[0,:])
        self.vector_of_cell_centered_grid_x = X_center.flatten()
        self.vector_of_cell_centered_grid_y = Y_center.flatten()
        self.vector_of_cell_centered_grid_z = Z_center.flatten()

        
        # ------------------------------------------------------------------

        # do the x edges

        #
        
        x_edge1 = ((np.arange(n1 - 1) + 0.5) * h1)[0, :]
        y_edge1 = ((np.arange(n2)) * h2)[0, :]
        z_edge1 = ((np.arange(n3)) * h3)[0, :]

        x_edge = []

        for ii in range(z_edge1.shape[0]):

            for jj in range(y_edge1.shape[0]):

                for kk in range(x_edge1.shape[0]):

                    x_edge.append([x_edge1[kk], y_edge1[jj], z_edge1[ii]])


        self.x_edges = np.asarray(x_edge)

        # ------------------------------------------------------------------

        # do the y edges

        #

        x_edge2 = ((np.arange(n1)) * h1)[0, :]
        y_edge2 = ((np.arange(n2 - 1) + 0.5) * h2)[0, :]
        z_edge2 = ((np.arange(n3)) * h3)[0, :]

        y_edge = []

        for ii in range(z_edge2.shape[0]):

            for jj in range(y_edge2.shape[0]):

                for kk in range(x_edge2.shape[0]):

                    y_edge.append([x_edge2[kk], y_edge2[jj], z_edge2[ii]])


        self.y_edges = np.asarray(y_edge)

        # ------------------------------------------------------------------

        # do the z edges

        #

        x_edge3 = ((np.arange(n1)) * h1)[0, :]
        y_edge3 = ((np.arange(n2)) * h2)[0, :]
        z_edge3 = ((np.arange(n3 - 1) + 0.5) * h3)[0, :]

        z_edges = []

        for ii in range(z_edge3.shape[0]):

            for jj in range(y_edge3.shape[0]):

                for kk in range(x_edge3.shape[0]):

                    z_edges.append([x_edge3[kk], y_edge3[jj], z_edge3[ii]])


        self.z_edges = np.asarray(z_edges)
        
        # ------------------------------------------------------------------

        # do the x faces

        #
        
        x_face1 = ((np.arange(n1 - 1) + 0.5) * h1)[0, :]
        y_face1 = ((np.arange(n2)) * h2)[0, :]
        z_face1 = ((np.arange(n3 - 1) + 0.5) * h3)[0, :]

        x_face = []

        for ii in range(z_face1.shape[0]):

            for jj in range(y_face1.shape[0]):

                for kk in range(x_face1.shape[0]):

                    x_face.append([x_face1[kk], y_face1[jj], z_face1[ii]])


        self.x_faces = np.asarray(x_face)
        
        # ------------------------------------------------------------------

        # do the y faces

        #
        
        x_face2 = (np.arange(n1) * h1)[0, :]
        y_face2 = ((np.arange(n2 - 1) + 0.5) * h2)[0, :]
        z_face2 = ((np.arange(n3 - 1) + 0.5) * h3)[0, :]
        
        y_faces = []

        for ii in range(z_face2.shape[0]):

            for jj in range(y_face2.shape[0]):

                for kk in range(x_face2.shape[0]):

                    y_faces.append([x_face2[kk], y_face2[jj], z_face2[ii]])


        self.y_faces = np.asarray(y_faces)
        
        # ------------------------------------------------------------------

        # do the z faces

        #
        
        x_face3 = ((np.arange(n1 - 1 ) + 0.5) * h1)[0, :]
        y_face3 = ((np.arange(n2 - 1) + 0.5) * h2)[0, :]
        z_face3 = (np.arange(n3) * h3)[0, :]

        z_faces = []

        for ii in range(z_face3.shape[0]):

            for jj in range(y_face3.shape[0]):

                for kk in range(x_face3.shape[0]):

                    z_faces.append([x_face3[kk], y_face3[jj], z_face3[ii]])


        self.z_faces = np.asarray(z_faces)       


    def getFaceDivergenceMatrix(self):
        """

            calculates the divergance operator on the faces

        """

        n1,n2,n3 = self.n1, self.n2, self.n3
        
        def ddx(n):
            return sp.spdiags((np.ones([n+1,1])*np.array([-1,1])).T,[0,1],n,n+1)
        
        D1 = sp.kron(sp.eye(n3),sp.kron(sp.eye(n2),ddx(n1)))
        D2 = sp.kron(sp.eye(n3),sp.kron(ddx(n2),sp.eye(n1)))
        D3 = sp.kron(ddx(n3),sp.kron(sp.eye(n2),sp.eye(n1)))

        #  DIV from faces to cell-centers
        Div = [D1, D2, D3]

        return sp.hstack(Div)


    def getNodalGradientMatrix(self):
        """

            calculates the gradient matrix

        """

        n1,n2,n3 = self.n1, self.n2, self.n3
        
        def ddx(n):
            return sp.spdiags((np.ones([n+1,1])*np.array([-1,1])).T,[0,1],n,n+1)
        
        G1 = sp.kron(sp.eye(n3+1), sp.kron(sp.eye(n2+1),ddx(n1)))
        G2 = sp.kron(sp.eye(n3+1), sp.kron(ddx(n2),sp.eye(n1+1)))
        G3 = sp.kron(ddx(n3),sp.kron(sp.eye(n2+1),sp.eye(n1+1)))
        # grad on the nodes
        Grad = sp.vstack([G1, G2, G3])

        return Grad
    
    def getEdgeCurlMatrix(self):
        """

            calculates the curl matrix operator

        """
        
        n1,n2,n3 = self.n1, self.n2, self.n3
        
        def ddx(n):
            return sp.spdiags((np.ones((n + 1, 1)) * [-1, 1]).T, [0, 1], n, n + 1, format="csr") # sp.spdiags((np.ones([n+1,1])*[-1,1]).T,[0,1],n,n+1)

        nfx = (n1 + 1) * n2 * n3
        nfy = n1 * (n2 + 1) * n3
        nfz = n1 * n2 * (n3 + 1)
        nex = n1 * (n2 + 1) * (n3 + 1)
        ney = (n1 + 1) * n2 * (n3 + 1)
        nez = (n1 + 1) * (n2 + 1) * n3
            
        Dyz = sp.kron(ddx(n3), sp.kron(sp.eye(n2), sp.eye(n1 + 1)))
        Dzy = sp.kron(sp.eye(n3), sp.kron(ddx(n2), sp.eye(n1 + 1)))
        
        Dxz = sp.kron(ddx(n3),sp.kron(sp.eye(n2+1),sp.eye(n1)))
        Dzx = sp.kron(sp.eye(n3),sp.kron(sp.eye(n2 + 1), ddx(n1)))
        
        Dxy = sp.kron(sp.eye(n3 + 1), sp.kron(ddx(n2),sp.eye(n1)))
        Dyx = sp.kron(sp.eye(n3+1), sp.kron(sp.eye(n2), ddx(n1)))

        # curl on the edges
        Curl = sp.vstack(
            [sp.hstack([sp.dia_matrix((nfx,nex)), Dyz, -Dzy]),
             sp.hstack([-Dxz, sp.dia_matrix((nfy,ney)), Dzx]),
             sp.hstack([Dxy, -Dyx  , sp.dia_matrix((nfz,nez))])
             ]
        )

        return Curl

    def getMeshGeometry(self):

        h1, h2, h3 = self.h1.T, self.h2.T, self.h3.T

        n1 = h1.size
        n2 = h2.size
        n3 = h3.size
        
        V = sp.kron(sp.diags(h3, [0]), sp.kron(sp.diags(h2, [0]), sp.diags(h1, [0])))
        
        F = sp.diags(sp.hstack([sp.kron(sp.diags(h3, [0]), sp.kron(sp.diags(h2, [0]), sp.eye(n1+1))).diagonal(),
             sp.kron(sp.diags(h3, [0]), sp.kron(sp.eye(n2 + 1), sp.diags(h1, [0]))).diagonal(),
             sp.kron(sp.eye(n3 + 1), sp.kron(sp.diags(h2, [0]),sp.diags(h1, [0]))).diagonal()]).toarray(), [0])
        
        L = sp.diags(sp.hstack([
            sp.kron(sp.eye(n3 + 1), sp.kron(sp.eye(n2 + 1), sp.diags(h1, [0]))).diagonal(),
            sp.kron(sp.eye(n3 + 1), sp.kron(sp.diags(h2, [0]), sp.eye(n1 + 1))).diagonal(),
            sp.kron(sp.diags(h3, [0]), sp.kron(sp.eye(n2 + 1), sp.eye(n1 + 1))).diagonal()]).toarray(), [0]
         )
        
        return V, F, L

    def getFaceToCellCenterMatrix(self):

        n1,n2,n3 = self.n1, self.n2, self.n3

        def av(n):
            return sp.spdiags((np.ones([n+1,1])*np.array([0.5,0.5])).T,[0,1],n,n+1)

        A1 = sp.kron(sp.eye(n3),sp.kron(sp.eye(n2),av(n1)))
        A2 = sp.kron(sp.eye(n3),sp.kron(av(n2),sp.eye(n1)))
        A3 = sp.kron(av(n3),sp.kron(sp.eye(n2),sp.eye(n1)))
        # average from faces to cell-centers
        Afc = sp.hstack([A1, A2, A3])

        return Afc

    def getEdgeToCellCenterMatrix(self):

        def av(n):
            return sp.spdiags((np.ones([n+1,1])*np.array([0.5,0.5])).T,[0,1],n,n+1)

        A1 = sp.kron(av(n3),sp.kron(av(n2),sp.eye(n1)))
        A2 = sp.kron(av(n3),sp.kron(sp.eye(n2),av(n1)))
        A3 = sp.kron(sp.eye(n3),sp.kron(av(n2),av(n1)))

        # average from edge to cell-centers
        Aec = sp.hstack([A1,A2,A3])

        return Aec

    def getNodalToCellCenterMatrix(self):

        def av(n):
            return sp.spdiags((np.ones([n+1,1])*np.array([0.5,0.5])).T,[0,1],n,n+1)

        Anc = sp.kron(av(n3), sp.kron(av(n2),av(n1)))

        return Anc

# plt.spy(mesh.getMeshGeometry())
# plt.show()
# print(sp.kron(sp.diags(h3.T, [0]), sp.kron(sp.eye(n2 + 1), sp.diags(h1.T, [0]))).diagonal())
#  sp.kron(sp.diags(h2.T, 1), sp.diags(h1.T, 1)))
