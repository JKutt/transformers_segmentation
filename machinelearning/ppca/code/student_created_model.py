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

        # Code for 1a) here for implementing the calculation of MLE
        # means, weights and varainces  

        raise NotImplementedError

        if self.verbose:
            print("eigenvectors:")
            print(eigenvecs)
            print(eigenvecs @ np.diag(lambdas) @ np.transpose(eigenvecs))
            print(f"Var ML: {self.sigma2 }")
            print(f"uq: {uq}")
            print(f"weights: {self.weights}")

    
    def transform(self, data):
        """

            latent from visible

        """

        [W, sigma, mean] = [self.weights, self.sigma2, self.mu]

        M = np.transpose(W).dot(W) + sigma * np.eye(self.q)
        Minv = np.linalg.inv(M)

        latent_data = Minv.dot(np.transpose(W)).dot(np.transpose(data - mean))
        latent_data = np.transpose(latent_data) 

        return latent_data

    
    def inverse_transform(self, data):
        """
        
            visible from latent
        
        """

        # calculate the tuned M
        M = np.transpose(self.weights).dot(self.weights) + self.sigma2 * np.eye(self.q)

        # create a simulation of the old data after beeing transformed with PCA
        created_data = self.weights.dot(
            np.linalg.inv(self.weights.T.dot(self.weights))
            ).dot(M).dot(data.T).T + self.mu

        return created_data


    def predict(self, Xtest):
        """
        
            prediction/reconstruction of input data
        
        """

        # reconstruct the inputs
        reduced_data = self.transform(Xtest)
        created_data = self.inverse_transform(reduced_data)

        return reduced_data, created_data
