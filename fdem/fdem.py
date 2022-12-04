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
        Grad = sp.hstack([G1, G2, G3])

        return Grad
    
    def getEdgeCurlMatrix(self):
        """

            calculates the curl matrix operator

        """
        
        n1,n2,n3 = self.n1, self.n2, self.n3
        
        def ddx(n):
            return sp.spdiags((np.ones([n+1,1])*np.array([-1,1])).T,[0,1],n,n+1)

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
        print(Dyz.shape, Dzy.shape, Dxz.shape, sp.eye(nfx,nex).shape)
        # curl on the edges
        Curl = sp.vstack(
            [sp.hstack([sp.eye(nfx,nex), Dyz, -Dzy]),
             sp.hstack([-Dxz, sp.eye(nfy,ney), Dzx]),
             sp.hstack([Dxy, -Dyx  , sp.eye(nfz,nez)])
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
            sp.kron(sp.diags(h3, [0]), sp.kron(sp.eye(n2 + 1), sp.eye(n1 + 1))).diagonal()])
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

    def getEdgeToCellCenterMatrix():

        def av(n):
            return sp.spdiags((np.ones([n+1,1])*np.array([0.5,0.5])).T,[0,1],n,n+1)
    
        A1 = sp.kron(av(n3),sp.kron(av(n2),sp.eye(n1)))
        A2 = sp.kron(av(n3),sp.kron(sp.eye(n2),av(n1)))
        A3 = sp.kron(sp.eye(n3),sp.kron(av(n2),av(n1)))
        
        # average from edge to cell-centers
        Aec = sp.hstack([A1,A2,A3])
        
        return Aec
    
    def getNodalToCellCenterMatrix():
        
        def av(n):
            return sp.spdiags((np.ones([n+1,1])*np.array([0.5,0.5])).T,[0,1],n,n+1)
            
        Anc = sp.kron(av(n3), sp.kron(av(n2),av(n1)))

        return Anc


n1=4
n2=5
n3=6
h1=10 * np.random.rand(4,1)
h2=10 * np.random.rand(5,1)
h3=10 * np.random.rand(6,1)

mesh = jmesh(n1, n2, n3, h1, h2, h3)
# print(sp.hstack([sp.kron(sp.diags(h3.T, [0]), sp.kron(sp.diags(h2.T, [0]), sp.eye(n1+1))).diagonal(),
#              sp.kron(sp.diags(h3.T, [0]), sp.kron(sp.eye(n2 + 1), sp.diags(h1.T, [0]))).diagonal(),
#              sp.kron(sp.eye(n3 + 1), sp.kron(sp.diags(h2.T, [0]),sp.diags(h1.T, [0]))).diagonal()]).toarray())
L = mesh.getMeshGeometry()

print(L)
# plt.spy(mesh.getMeshGeometry())
# plt.show()
# print(sp.kron(sp.diags(h3.T, [0]), sp.kron(sp.eye(n2 + 1), sp.diags(h1.T, [0]))).diagonal())
#  sp.kron(sp.diags(h2.T, 1), sp.diags(h1.T, 1)))
