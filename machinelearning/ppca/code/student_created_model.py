import numpy as np
import scipy.linalg as sl

class ppca:
    """
    
        Model for Probabilisitc PCA using the MLE method
    
    """

    def __init__(self, X=None, num_components=1, verbose=False):

        # set parameters
        self.q = num_components
        self.d = X.shape[1]
        self.mu = None
        self.sigma2 = None
        self.weights = None
        self.verbose = verbose

        if X is not None:
            self.fit(X)


    def fit(self, X):
        """
        
            Calculate the MLE
        
        """

        # calculate maximum likelihood means
        self.mu = np.mean(X, axis=0)

        # Calcualte maximum likelihood variance
        data_cov = np.cov(X, rowvar=False)
        lambdas, eigenvecs = np.linalg.eig(data_cov)
        idx = lambdas.argsort()[::-1]   
        lambdas = lambdas[idx]
        eigenvecs = - eigenvecs[:,idx]

        self.sigma2 = (1.0 / (self.d-self.q)) * sum([lambdas[j] for j in range(self.q, self.d)])

        # Calculate maximum likelihood Weight matrix
        uq = eigenvecs[:,:self.q]
        lambdaq = np.diag(lambdas[:self.q])
        
        self.weights = uq @ np.sqrt(lambdaq - self.sigma2 * np.eye(self.q))

        if self.verbose:
            print("eigenvectors:")
            print(eigenvecs)
            print(eigenvecs @ np.diag(lambdas) @ np.transpose(eigenvecs))
            print(f"Var ML: {self.sigma2 }")
            print(f"uq: {uq}")
            print(f"weights: {self.weights}")

    
    def transform(self, data):
        """

            hidden from visible

        """

        # write the code here that samples the hidden/latent space

        raise NotImplementedError()

    
    def inverse_transform(self, data):
        """
        
            visible from hidden
        
        """

        # write the reconstruction code here

        raise NotImplementedError()


    def predict(self, Xtest):
        """
        
            prediction/reconstruction of input data
        
        """

        # reconstruct the inputs
        reduced_data = self.transform(Xtest)
        created_data = self.inverse_transform(reduced_data)

        return reduced_data, created_data
