---
# Math frontmatter:
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

# Geological Segmentation-Guided Regularization

## Summary

Geological structure is distinctive and boundaries are sharp contacts between units. Geophysical models on the other hand are often smooth and geological meaning is interpreted. These interpretations are often subjective and the model itself can be unconstrained and not always influenced by prior information. Presenting these results to a non geophysicist can be tough. Typically delineations are sketched or overlain the geophysical model in order to fully communicate the results. By influencing the geophysical model with prior information like petrophysical data, we can build quasi-geological models from the geophysics. My research goal of creating quasi-geological models within the inversion process is to provide a product that better communicates results to the non-geophysicist. Quasi-geological models can provide well defined, sharp boundaries representing geological rock units. This is more closely in tune with the language of resource explorers who employ geophysical methods to help target drill holes. We can then more clearly answer questions about targeting interfaces like unconformities or intrusive units which are often associated with mineralization zones.


## Introduction

Geophysicists interpret inversion models in order to communicate the results to the non-geophysicists. Questions from geologists we hope to answer:

1) Structural dip information.

2) Interfaces. 

3) Volume estimates.

These questions can be a challenge to anwser because geophysical models are non-unique and ill-posed. The problem is an optimization problem solved by minimizing the objective function ([Tikhonov and Arsenin, 1977](https://www.scirp.org/(S(351jmbntvnsjt1aadkozje))/reference/referencespapers.aspx?referenceid=1111962)):

$$
\label{reginv}
\begin{aligned}
\underset{\mathbf{m}}{min} \; \phi(\mathbf{m}) = \phi_{d}(\mathbf{m}) + \beta\phi_{m}(\mathbf{m})
\end{aligned}
$$

Where
$$
\label{reginv}
\begin{aligned}
\phi_{d}(\mathbf{m}) = \frac{1}{2} \| \mathbf{W}_d \left( \mathcal{F}(\mathbf{m}) - \mathbf{d}_{obs} \right) \|^{2}
\end{aligned}
$$

$$
\label{reginv}
\begin{aligned}
\phi_{m}(\mathbf{m}) = \frac{1}{2} \| L (\mathbf{m} - \mathbf{m}_{ref}) \|^{2}
\end{aligned}
$$

Here $\mathbf{m}$ is the model, $\mathbf{d}_{obs}$ are the observed data, $\mathbf{W}_d$ denoted the data weights, $\mathcal{F}(\mathbf{m})$ represents the forward operator and $L$ denotes the regularization operator. To model objective function is used to stabalize the solution and allows prior information to be encoded into the inversion to promote user defined structure. A common regularization for geophysical problems is ([](https://doi.org/10.1190/1.1443692)):

$$
\label{reginv}
\begin{aligned}
\phi_m(\mathbf{m}) = \phi_{small}(\mathbf{m}) + \phi_{smooth}(\mathbf{m})
\end{aligned}
$$

where expanded, represents:

$$
\label{reginv}
\begin{aligned}
\phi_{m}(\mathbf{m}) =\alpha_s \int_V \bigl\{ m(\vec{r}) - m_0 \bigr\}^2 dv \\ + \alpha_x \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{x}} \Bigr\}^2 dv \\ + \alpha_y \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{y}} \Bigr\}^2 dv \\ + \alpha_z \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{z}} \Bigr\}^2 dv
\end{aligned}
$$

Segmentations.....

## Rotated Gradients

Rotating the objective function ([](https://doi.org/10.1190/1.1444705))

$$
\label{reginv}
\begin{aligned}
\phi_{m}(\mathbf{m}) =\alpha_s \int_V \bigl\{ m(\vec{r}) - m_0 \bigr\}^2 dv \\ + \alpha_{x'} \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{x'}} \Bigr\}^2 dv \\ + \alpha_{y'} \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{y'}} \Bigr\}^2 dv \\ + \alpha_{z'} \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{z'}} \Bigr\}^2 dv
\end{aligned}
$$

## Geological Segmentation with Transformers

## Geological Classification

## Segmentation-guided regularization