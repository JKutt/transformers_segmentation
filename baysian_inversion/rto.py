import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from discretize import TensorMesh
from SimPEG import simulation, maps
import numpy as np
import matplotlib.pyplot as plt


def randomize_them_optimize(
        
        dobs:np.ndarray,
        G:np.ndarray,
        n_samples:int=50,
        n_model_samples:int=100,
        n_components:int=20,
        mu_init:float=1e1,
        mu_perturb:float=1e4,
        standard_deviation:np.ndarray=None,
        
    ) -> np.ndarray:
    """
    
        runs rto on a simulation
    
    """

    # Generate sample data
    np.random.seed(0)

    # data covariace matrix to perturb data
    Wd = np.diag(standard_deviation)

    # model zero mean guassian distribution
    identity_matrix = np.eye(n_model_samples)
    zero_means = np.zeros(n_model_samples)

    draws = []

    # draw samples of the posterior
    for ii in range(n_samples):
    
        # draw from perturbed data distribution
        perturbed_data = np.random.multivariate_normal(dobs, Wd, size=1)

        # draw from perturbed model distribution
        s = np.random.multivariate_normal(zero_means, identity_matrix, size=1)

        # coefficient matrix
        Wm = np.sqrt(mu_perturb) * np.eye(n_model_samples)
        
        # solve linear equation for perturbed model
        perturbed_model = np.linalg.solve(Wm, s.T)

        # data covariance matrix
        Cd = np.diag(1 / standard_deviation)

        # solve foe model
        A = G.T @ Cd.T @ Cd @ G + mu_init * np.eye(G.shape[1])
        b = G.T @ Cd.T @ Cd @ perturbed_data.T + mu_init * np.eye(G.shape[1]) @ perturbed_model
        m_max_post = np.linalg.solve(A, b)
        draws.append(m_max_post)

    return np.hstack(draws)

def run():

    # Number of data observations (rows)
    n_data = 20

    n_param = 100  # Number of model paramters

    # A 1D mesh is used to define the row-space of the linear operator.
    mesh = TensorMesh([n_param])

    # Creating the true model
    true_model = np.zeros(mesh.nC)
    true_model[mesh.cell_centers_x > 0.3] = 1.0
    true_model[mesh.cell_centers_x > 0.45] = -0.5
    true_model[mesh.cell_centers_x > 0.6] = 0

    # Create the linear operator for the tutorial.
    # The columns of the linear operator represents a set of decaying and oscillating functions.
    sim = simulation.ExponentialSinusoidSimulation(
        model_map=maps.IdentityMap(), mesh=mesh, n_kernels=n_data, p=-0.25, q=0.25, j0=1, jn=60
    )

    # Standard deviation of Gaussian noise being added
    data_std = 0.001
    np.random.seed(3211)

    # Create a SimPEG data object
    data_obj = sim.make_synthetic_data(true_model, noise_floor=data_std, add_noise=True)

    # run rto
    rto_results = randomize_them_optimize(
        
        data_obj.dobs,
        sim.G,
        n_samples=1000,
        n_model_samples=n_param,
        n_components=n_data,
        mu_init=1e1,
        mu_perturb=1e4,
        standard_deviation=data_obj.standard_deviation,
        
    )

    print(rto_results.shape)
    plt.plot(true_model, label='true model')
    plt.plot(rto_results.mean(axis=1), 'g', label='RTO')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    run()