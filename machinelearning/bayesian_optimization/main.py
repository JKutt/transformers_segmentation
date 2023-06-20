import numpy as np
import torch
from matplotlib import pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import SGD

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f' Using device: {device}')
dtype = torch.float

# ----------------------------------------------------------------------------------------

# get our data

#

def getTime():
    timeFrom = [2040., 2060., 2080., 2120., 2160., 2200.,
                2240., 2320., 2400.,
                2480., 2560., 2640.,
                2720., 2800., 2960.,
                3120., 3280., 3440.,
                3600., 3760.]
    timeTo = [2060., 2080., 2120., 2160., 2200., 2240.,
              2320., 2400., 2480., 2560., 2640., 2720.,
              2800., 2960., 3120., 3280., 3440.,
              3600., 3760., 3920.]
    return timeFrom, timeTo


timeFrom, timeTo = getTime()

mid_time = (np.asarray(timeTo) + np.asarray(timeFrom)) / 2
time_width = np.asarray(timeTo) - np.asarray(timeFrom)

secondary_voltage = [
                    -229.8557,
                    -210.5083,
                    -195.3468,  
                    -177.7571,  
                    -162.1708,   
                    -149.5160,  
                    -137.7064,  
                    -127.3763,  
                    -118.6991, 
                    -109.4883, 
                    -101.7745,  
                    -95.0854, 
                    -86.8954,
                    -79.6953, 
                    -72.7827, 
                    -66.0279,  
                    -60.5468, 
                    -55.2554, 
                    -50.1276, 
                    -47.6666,
]

secondary_voltage = torch.from_numpy(-1 * np.array(secondary_voltage))

secondary_voltage = secondary_voltage.to(device)

# ### Set up function to model
# In this tutorial we will model a simple sinusoidal function with i.i.d. Gaussian noise:
# 
# $$y = \sin(2\pi x) + \epsilon, ~\epsilon \sim \mathcal N(0, 0.15)$$

# #### Initialize training data


# use regular spaced points on the interval [0, 1]
# train_X = torch.linspace(0, 1, 15, dtype=dtype, device=device)

train_X = torch.from_numpy(mid_time) * 1e-3
train_X = train_X.to(device)
# training data needs to be explicitly multi-dimensional
train_X = train_X.unsqueeze(1)

# sample observed values and add some synthetic noise
# train_Y = torch.sin(train_X * (2 * np.pi)) + 0.15 * torch.randn_like(train_X)
train_Y = secondary_voltage
train_Y = train_Y.unsqueeze(1)


# #### Initialize the model
# We will model the function using a `SingleTaskGP`, which by default uses a `GaussianLikelihood` and infers the unknown noise level.
# 
# The default optimizer for the `SingleTaskGP` is L-BFGS-B, which takes as input explicit bounds on the noise parameter. However, the `torch` optimizers don't support parameter bounds as input. To use the `torch` optimizers, then, we'll need to manually register a constraint on the noise level. When registering a constraint, the `softplus` transform is applied by default, enabling us to enforce a lower bound on the noise.
# 
# **Note**: Without manual registration, the model itself does not apply any constraints, due to the interaction between constraints and transforms. Although the `SingleTaskGP` constructor does in fact define a constraint, the constructor sets `transform=None`, which means that the constraint is not enforced. See the [GPyTorch constraints module](https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/constraints/constraints.py) for additional information.
# 


model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))


# #### Define marginal log likelihood 
# We will jointly optimize the kernel hyperparameters and the likelihood's noise parameter, by minimizing the negative `gpytorch.mlls.ExactMarginalLogLikelihood` (our loss function).

mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
# set mll and all submodules to the specified dtype and device
mll = mll.to(train_X)


# #### Define optimizer and specify parameters to optimize
# We will use stochastic gradient descent (`torch.optim.SGD`) to optimize the kernel hyperparameters and the noise level. In this example, we will use a simple fixed learning rate of 0.1, but in practice the learning rate may need to be adjusted.
# 
# Notes:
# - As the `GaussianLikelihood` module is a of child (submodule) of the `SingleTaskGP` moduel, `model.parameters()` will also include the noise level of the `GaussianLikelihood`. 
# - A subset of the parameters could be passed to the optimizer to tune those parameters, while leaving the other parameters fixed.

optimizer = SGD([{"params": model.parameters()}], lr=0.01)


# #### Fit model hyperparameters and noise level
# Now we are ready to write our optimization loop. We will perform 150 epochs of stochastic gradient descent using our entire training set.

NUM_EPOCHS = 100

model.train()

for epoch in range(NUM_EPOCHS):
    # clear gradients
    optimizer.zero_grad()
    # forward pass through the model to obtain the output MultivariateNormal
    output = model(train_X)
    # Compute negative marginal log likelihood
    loss = -mll(output, model.train_targets)
    # back prop gradients
    loss.backward()
    # print every 10 iterations
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
            f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():>4.3f} "
            f"noise: {model.likelihood.noise.item():>4.3f}"
        )
    optimizer.step()


# #### Compute posterior over test points and plot fit
# We plot the posterior mean and the 2 standard deviations from the mean.
# 
# Note: The posterior below is the posterior prediction for the underlying sinusoidal function, i.e., it does not include the observation noise. If we wanted to get the posterior prediction for the observations (including the predicted observation noise), we would instead use `posterior = posterior = model.posterior(test_X, observation_noise=True)`. 

# set model (and likelihood)
model.eval()
# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(6, 4))
# test model on 101 regular spaced points on the interval [0, 1]
test_X = torch.linspace(2, 4, 101, dtype=dtype, device=device)
# no need for gradients
with torch.no_grad():
    # compute posterior
    posterior = model.posterior(test_X)
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()
    # Plot training points as black stars
    ax.plot(train_X.cpu().numpy(), train_Y.cpu().numpy(), "k*")
    # Plot posterior means as blue line
    ax.plot(test_X.cpu().numpy(), posterior.mean.cpu().numpy(), "b")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(
        test_X.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5
    )
ax.legend(["Observed Data", "Mean", "Confidence"])
plt.tight_layout()
plt.show()
