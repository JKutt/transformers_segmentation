---
# Math frontmatter:
math:
  '\mariginal' : '\mathcal{P}(\mathbf{d} | \mathbf{m})'
  '\posterior' : '\mathcal{P}(\mathbf{m} | \mathbf{d})'
  '\prior'     : '\mathcal{P}(\mathbf{m}'
  '\fm'        : '\mathcal{F}(\mathbf{m})'
  '\d'         : '\mathbf{d}'
  '\m'         : '\mathbf{m}'
  '\wd'         : '\mathbf{W}_d'
  '\dp'         : '\tilde{\mathbf{d}}'
  '\mp'         : '\tilde{\mathbf{m}}'

---

# Title Ideas

- **Uncertainty Quantification of Regularizations in Bayesian Geophysical Inversions**
- **Using Uncertainty Quantification to Measure Regularizations in Bayesian Geophysical Inversions**

## introduction
Typical deterministic geophysical models negate the entire model space and consider a single objectively decided likeliest model. However, the geophysical inverse problem is ill conditioned meaning many solutions exist. Uncertainty quantification can allow us to examine the entire soltuion space giving us a confidence criteria for a given model (2014 paper ref in Blatter). We can deduce that the solution space is dependent on the regularization used. In the bayesian approach the prior is our choice of regularization. The choice of prior will then influence the uncertainty quantification. By adjusting the regularization we can hope to recover better estimates on depth to interfaces or unit structure. In addition we can expect better recovered physical property estimates. This can be benificial in when looking at depths to bed rock unconformities or the dip of a geological unit. By using Bayesian inversion framework to calculate a posterior that can be sampled from we can easily substitute in different regularizations and sampling methids to generate uncertainty quantification used to interpret the recovered models. 

## Posing Geophysical inversion in probabilistic terms
$$
\posterior \propto \marginal \prior
$$

Relating this to the geophysical problem:

$$
\marginal \propto \exp \left( -\frac{1}{2} \| \wd \left( \fm - \d \right) \|^{2} \right)
$$

$$
\prior \propto \exp \left( -frac{\beta}{2} \| L \m \right)
$$

where $\m$ is the model parameters vector, $\d$ is the data vector, $\fm$ is the predicted data givem a model $\m$, $\wd$ is the noise covariance matrix of the measured data, L represents the regularization opertor and $\beta$ is essentially the trade-off parameter.

We can then sample from the posterior distribution by finding the minimum to the negative log-likelihood of $\posterior$

## Uncertainty Quantification
- Keep general can include MCMC

## Regularization Uncertainty Quantification
