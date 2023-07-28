---
# Math frontmatter:
title: My PDF
exports:
  - format: pdf
    template: arxiv_two_column
    output: ./figures/document.pdf
math:
  '\marginal' : '\mathcal{P}(\mathbf{d} | \mathbf{m})'
  '\posterior' : '\mathcal{P}(\mathbf{m} | \mathbf{d})'
  '\prior'     : '\mathcal{P}(\mathbf{m})'
  '\mariginalp' : '\mathcal{P}(\tilde{\mathbf{d}} | \mathbf{m})'
  '\posteriorp' : '\mathcal{P}(\tilde{\mathbf{m}} | \tilde{\mathbf{d}})'
  '\priorp'     : '\mathcal{P}(\tilde{\mathbf{m}})'
  '\marginalu' : '\mathcal{P}(\mathbf{d} | \mathbf{m}, \mu)'
  '\posterioru' : '\mathcal{P}(\mathbf{m}, \mu | \mathbf{d})'
  '\prioru'     : '\mathcal{P}(\mathbf{m} | \mu)'
  '\priormu'    : '\mathcal{P}(\mu)'
  '\fm'        : '\mathcal{F}(\mathbf{m})'
  '\fmp'        : '\mathcal{F}(\tilde{\mathbf{m}})'
  '\d'         : '\mathbf{d}'
  '\m'         : '\mathbf{m}'
  '\wd'         : '\mathbf{W}_d'
  '\dp'         : '\tilde{\mathbf{d}}'
  '\mp'         : '\tilde{\mathbf{m}}'

---

# Title Ideas

- **Uncertainty Quantification of Regularizations in Bayesian Geophysical Inversions**

- **Using Uncertainty Quantification to Measure Regularizations in Bayesian Geophysical Inversions**

- **Using Randomize then Optimize to Measure Regularizations in Bayesian Geophysical Inversions**

- **Using Randomize then Optimize to Measure Regularizations in Bayesian Electromagnetic Inversions**

## Abstract

Deterministic geophysical inversion often negates the entire model space and produces a single maximum likelihood model where one model is then objectively decided for the interpretation. However, the geophysical inverse problem is non-unique meaning many solutions exist. Uncertainty quantification on the other hand allows us to examine the entire solution space giving us a confidence criteria for a given result. Regularization plays an important role in shaping the solution space as it penalizes or promotes certain models. In deterministic inversion, uncertainty estimates can be obtained from regularized methods via a local linearization comparing a pre selected reference model ([](https://doi.org/10.1137/1.9780898717921)). Others regularized methods explore the model space via norms ([](https://doi.org/10.1093/gji/ggz156)). These methods can constrain the choice of models for various effect be it sharpening boundaries, compacting or stretching features. In the Bayesian approach the prior is our choice of regularization and the choice of prior will then influence the uncertainty estimates in a non-linear way. By adjusting the regularization we can hope to recover better estimates for depth to interfaces or overall sharper recovered models. In addition we can expect better recovered physical property estimates. This is particularly important when using DC-resistivity and magnetotellurics to determine depth to bedrock unconformities or the dips and geometries of a geological unit. By using the probabilistic interpretation of a regularized inversion we can calculate a posterior where Bayesian sampling methods are used to recover models. This allows us to easily substitute in different regularizations and sampling methods to generate uncertainty information used to interpret the recovered models.

To calculate the posterior we first use a randomize then optimize (RTO) approach by generating a perturbed model and data distributions ([](https://doi.org/10.1137/140964023)). By perturbing both an uncertainty can be extracted by sampling from posterior giving us solutions using the perturbed inputs. The objective function can be represented in a probabilistic form using Bayes Theorem:

$$
\posterior \propto \marginal \prior
$$

where the marginal and prior are:

$$
\marginal \propto \exp \left( -\frac{1}{2} \| \wd \left( \fm - \dp \right) \|^{2} \right)
$$

$$
\prior \propto \exp \left( -\frac{\beta}{2} \| L (\m - \mp) \|^{2} \right)
$$

giving us the minimizer:

$$
\label{reginv}
\begin{aligned}
\underset{\m}{min} \; f(\m) = \frac{1}{2} \| \wd \left( \fm - \dp \right) \|^{2} + \frac{\beta}{2} \| L (\m - \mp) \|^{2}
\end{aligned}
$$

where $\m$ is the model parameters vector, $\d$ is the data vector, $\fm$ is the predicted data given a model $\m$, $\wd$ is the noise covariance matrix of the measured data, L represents the regularization operator, $\dp \sim \mathcal{N}(\d, \sqrt{\wd})$ is the perturbed data, $\mp \sim \mathcal{N}(0, \frac{1}{\beta}\left( L^T L\right)^{-1})$ is the perturbed model and $\beta$ is the trade-off parameter for the regularization. 

We can explore the model space by drawing samples $\m^i$ where $i=1,...,N_{samples}$ via Gibbs sampling; $i$ representing the Gibbs sampling iteration. Within the Gibbs sampling steps there are many choices of sampler techniques to use such as Markov chain Monte Carlo and Metropolis-Hastings. These in particular perform an accept/reject at every iteration of the Markov chain. This can be computationally expensive as the number of samples gets large. That is due to the serial loop in the algorithms where the $i^{th}$ iteration depends on $i - 1$. These methods though are proven to converge and are reliable. 

In the work of [](https://doi.org/10.1093/gji/ggac241) all RTO samples are accepted resulting in a biased distribution, but shown to be still a good approximation of the Bayesian posterior for electromagnetic problems. By accepting all samples, every iteration is independent of each other and the problem becomes highly parallelizable and efficient. Furthermore, the RTO algorithm can be modified to provide a way to calculate a distribution for the tradeoff parameter $\beta$ by treating it as an unknown to get a posterior $\posterioru$ called the RTO-TKO method ([](https://doi.org/10.1093/gji/ggac241) Part I and [](https://doi.org/10.1093/gji/ggac242) Part II). In this case every Gibbs iteration produces $\m^{i+1}, \mu^{i + 1}$ pairs of samples. The RTO-TKO algorithm provides yet another control on the regularization by allowing a choice of the prior $\priormu$.

Beginning with a linear problem we can compare the results of the RTO and RTO-TKO to a typical deterministic inversion. In [](#linearmod) the RTO and RTO-TKO results show all the models that fall within the 95th percentile. In [](#betadist) shows us the distribution of the most frequent $\beta$'s that give us optimal results. From the following results we get a $\phi_d^{RTO}=37.04$ with set $\beta=10$ and $\phi_d^{RTO-TKO}=16.79$ using the mean of the $\beta$ distribution $\beta=3.76$. Our $\phi_d^*=20$ for this linear model case. The model norms are $\phi_m^{RTO}=323.31$ and $\phi_m^{RTO}=362.59$ respectively.

```{figure} ./figures/linear_models_result.png
:height: 350px
:width: 450px
:name: linearmod
:alt: linear model results
:align: center


Recovered models for RTO, RTO-TKO and deterministic methods
```

```{figure} ./figures/rto-tko_beta_dist_linear_model.png
:height: 350px
:width: 450px
:name: betadist
:alt: linear model results
:align: center


Distribution of trade off parameter $\beta$
```

Using the Bayesian inversion approach we can explore a model space for various regularizations comparing the uncertainty quantification between each. In addition, with RTO-TKO we can further probe the regularization choices by analyzing the regularization $\beta$ distribution. These distributions often have large variance and multiple peaks. Exploring the differences in models given different high probability $\beta$'s can give much needed insight into uncertainty in our recovered interfaces for the DC-resistivity and magnetotellurics problems. Both 1-D and 2-D simulations using the RTO method as a base will be generated and compared to traditional geophysical inversion results in order to illustrate the improvement in interpretation. 



