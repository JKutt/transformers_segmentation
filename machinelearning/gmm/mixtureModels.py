import imageio
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image
from sklearn import datasets
from sklearn.cluster import KMeans
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
import copy
from pymatsolver import PardisoSolver
from scipy.stats import norm
import discretize
import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth
from typing import List, Iterable


def torch_gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    return torch.diagonal(1 / ((2 * np.pi) ** (n / 2) * torch.linalg.det(cov) ** 0.5) * np.exp(-0.5 * torch.matmul(torch.matmul(diff.T, torch.linalg.inv(cov)), diff)))


class torchGaussianMixtureModel(nn.module):
    """
    
        pyTorch implimentation of Gaussian mixture model

    """

    def __init__(self, number_of_clusters: int) -> None:
        super().__init__()

        self.clusters = []
        self.gamma_nk = None
        self.totals = None
        self.n_clusters = number_of_clusters

        # TODO: determine proportions
        

    def expectation_step(self):

        N = self.X.shape[0]
        K = len(self.clusters)
        self.totals = torch.zeros(N, 1)
        self.gamma_nk = torch.zeros(N, K)
        
        for k, cluster in enumerate(self.clusters):
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']
        
            self.gamma_nk[:, k] = (pi_k * torch_gaussian(self.X, mu_k, cov_k)).ravel()
        
        self.totals = torch.sum(self.gamma_nk, 1)
        self.gamma_nk /= torch.unsqueeze(self.totals, 1)

    def maximization_step(self):

        N = float(self.X.shape[0])
    
        for k, cluster in enumerate(self.clusters):
            gamma_k = torch.unsqueeze(self.gamma_nk[:, k], 1)
            N_k = torch.sum(self.gamma_k, axis=0)
            
            pi_k = N_k / N
            mu_k = torch.sum(gamma_k * self.X, axis=0) / N_k
            cov_k = (gamma_k * (self.X - mu_k)).T @ (self.X - mu_k) / N_k
            
            cluster['pi_k'] = pi_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k

    def get_likelihood(self):

        sample_likelihoods = torch.log(self.totals)
        
        return torch.sum(sample_likelihoods), sample_likelihoods
    
    def fit(self, X, n_epochs):

        # We use the KMeans centroids to initialise the GMM    
        kmeans = KMeans(self.n_clusters).fit(X)
        mu_k = kmeans.cluster_centers_
        
        for i in range(self.number_of_clusters):
            self.clusters.append({
                'pi_k': 1.0 / self.number_of_clusters,
                'mu_k': torch.from_numpy(mu_k[i]),
                'cov_k': torch.eye(X.shape[1])
            })

        likelihoods = torch.zeros(n_epochs, )
        scores = torch.zeros(X.shape[0], self.n_clusters)
        history = []

        for i in range(n_epochs):
            clusters_snapshot = []
            
            # This is just for our later use in the graphs
            for cluster in self.clusters:
                clusters_snapshot.append({
                    'mu_k': cluster['mu_k'],
                    'cov_k': cluster['cov_k']
                })
                
            history.append(clusters_snapshot)
        
            self.expectation_step(X)
            self.maximization_step(X)

            likelihood, sample_likelihoods = self.get_likelihood(X)
            likelihoods[i] = likelihood

            print('Epoch: ', i + 1, 'Likelihood: ', likelihood)

        scores = torch.log(self.gamma_nk)
        
        return self.clusters, likelihoods, scores, sample_likelihoods, history
    
    def predict(self, y):

        samples = []

        for i in range(y.shape[0]):
            # sample uniform
            r = np.random.uniform(0, 1)
            # select gaussian
            k = 0
            for i, threshold in enumerate(acc_pis):
                if r < threshold:
                    k = i
                    break

            selected_mu = mus[k]
            selected_cov = covs[k]

            # sample from selected gaussian
            lambda_, gamma_ = np.linalg.eig(selected_cov)

            dimensions = len(lambda_)
            # sampling from normal distribution
            y_s = np.random.uniform(0, 1, size=(dimensions * 1, 3))
            x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
            # transforming into multivariate distribution
            x_multi = (x_normal * lambda_) @ gamma_ + selected_mu
            samples.append(x_multi.tolist()[0])