---
# Math frontmatter:
math:

---

# Geological Segmentation-Guided Regularization

## Summary

Geological structure is distinctive and boundaries are sharp contacts between units. Geophysical models on the other hand are often smooth and geological meaning is interpreted. These interpretations are often subjective and the model itself can be unconstrained and not always influenced by prior information. Presenting these results to a non geophysicist can be tough. Typically delineations are sketched or overlain the geophysical model in order to fully communicate the results. By influencing the geophysical model with prior information like petrophysical data, we can build quasi-geological models from the geophysics. My research goal of creating quasi-geological models within the inversion process is to provide a product that better communicates results to the non-geophysicist. Quasi-geological models can provide well defined, sharp boundaries representing geological rock units. This is more closely in tune with the language of resource explorers who employ geophysical methods to help target drill holes. We can then more clearly answer questions about targeting interfaces like unconformities or intrusive units which are often associated with mineralization zones.


## Introduction

Geophysicists interpret inversion models in order to communicate the results to the non-geophysicists. Questions from geologists we hope to answer:

1) Structural dip information.

2) Interfaces. 

3) Volume estimates.

These questions can be a challenge to anwser because geophysical models are non-unique and ill-posed. The problem is an optimization problem solved by minimizing an objective function ([Tikhonov & Arsenin (1977)](https://www.scirp.org/(S(351jmbntvnsjt1aadkozje))/reference/referencespapers.aspx?referenceid=1111962)). within this framework there has been multiple approaches to produce inversion results better that are similar to typical geological structure. Early approaches used a constrained inversions ([](https://dx.doi.org/10.14288/1.0052682), [](https://dx.doi.org/10.14288/1.0052390)) to introduce strong a priori information and bounds to the accepted models. This information can be encoded in either the reference model or smalness weights. Most common information is from drill logs including measurments made in-situ. Others explored different model norms ([](10.1093/gji/ggu067), [](10.1093/gji/ggz156), [](https://doi.org/10.1111/1365-2478.13063)). By using sparse norms more compact bodies are promoted in the model space giving sharper defined interfaces. This is commony used today and lead into more methods exploring the model regularization to explore recovering dips of structures ([](https://doi.org/10.1111/1365-2478.13417))and compact geological structures.

Recent approaches integrate geologic structure by creating a quasi-geoogical model from the geological inversion ([](10.1093/gji/ggz389), [Balza et al. (2023)](https://ui.adsabs.harvard.edu/abs/2022AGUFMNS34B..01B), [](10.1190/INT-2019-0272.1)). Petrophyically and Geologically Guided Inversion (PGI) encodes a Guassian mixture model with physical properties within the smallness term of the model regularization.

Taking things further with the PGI framework, Astic then introduces image segmentation methods for thee geological classification. By swapping out the Gaussian mixture model (GMM) with Gaussian mixture Markov Random field (GMMRF) and a coupling matrix that encodes geological rules([](10.1190/segam2021-3583615.1)) better quasi-geological models are prodcued. The GMMRF This was used to solve the onion problem with the standard Gaussian mixture model ([](https://doi.org/10.14288/1.0394725)). The GMMRF has the benifit of incorporating spatial information via a defined neighborhood of a single cell. This is the connection for where the geological rules can be enforced. By accessing different neighborhoods, geological orientations can be infered.

Keeping with the theme of image segmentation for the geological classification, the GMMRF is a manual approach with hyper parameters that may or may not work for every model. Recent success with machine earning models such as the transformer where at there core take advantage of the construct of the attention mechanism ([](https://doi.org/10.48550/arXiv.1706.03762)). Attention is unique in as it learns to identify pixels (or cells) and classifies them according to the unique set of neighbouring pixels often refered to as heat maps. Transformers, though began in natural language processing, have recently been successful in the computer vision space ([](https://doi.org/10.48550/arXiv.2010.11929), [](https://doi.org/10.48550/arXiv.2304.02643)). Recently transformers have been applied to medical imagining probems where segmentation is required for identifying stuctures in the brain ([](https://doi.org/10.48550/arXiv.2306.11730)). This paper will explore the use of Transformers for the segmentation of the geophysical model in order to infer the geological structures and their orientations. This information will then be provided to both regularization components the smallness and smoothness. The segmentations will give us the hard boundaries while the orientations and geological boundaries can be used to guide the regularization of our objects. The contribution of this work is to build up a quasi-geological model that can produce a prior taht drives the inversion to explore models more closely resembling geological structure and contacts. 

## Rotated Gradients

Starting with our objective function we want to minimize represented as the following

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

Rotating the objective function ([](https://doi.org/10.1190/1.1444705))

$$
\label{reginvrot}
\begin{aligned}
\phi_{m}(\mathbf{m}) =\alpha_s \int_V \bigl\{ m(\vec{r}) - m_0 \bigr\}^2 dv \\ + \alpha_{x'} \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{x'}} \Bigr\}^2 dv \\ + \alpha_{y'} \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{y'}} \Bigr\}^2 dv \\ + \alpha_{z'} \int_V \Bigl\{ \frac{\partial{[m(\vec{r}) - m_0]}}{\partial{z'}} \Bigr\}^2 dv
\end{aligned}
$$

The derivatives obtained have directional vectors in ($\hat{x}$, $\hat{y}$, $\hat{z}$):

$$
\label{reginv}
\begin{aligned}
\vec{g}_s = \mathbf{R} \vec{g}_u
\end{aligned}
$$

Where $\mathbf{R}$ is the rotation matrix taking the gradient vectors from ($x$, $y$, $z$) coordinate frame to the ($x'$, $y'$, $z'$) rotated frame. If we substitute this into [](#reginvrot) we can organize the objective function for the smoothness portion to be:

$$
\label{regrot}
\begin{aligned}
\phi_{smoothness}(\mathbf{m}) = \int_{V} \nabla \mathbf{m} \cdot \mathbf{a} \nabla \mathbf{m} \;dV
\end{aligned}
$$

$$
\label{reginv}
\begin{aligned}
\mathbf{a} = \mathbf{Q} \mathbf{L} \mathbf{Q}^{-1}
\end{aligned}
$$

Where $\mathbf{Q}$ are the regularization directions and $\mathbf{L}$ are the directional $\alpha$'s. Now with tensor **a** we can influence the regularization direction by defining the reference axis and $\alpha$'s for each cell. In [](#rotated_gradients) we introduce the prior knowledge expecting a 45 degree westerly dip structure and apply the rotation to the objective function within the define structure and unrotated everywhere else.

```{figure} ./figures/rotated_gradients.png
:height: 350px
:width: 550px
:name: rotated_gradients
:alt: gradrotate
:align: center


Recovered models and their distributions a) the true model, b) Tikhonov result, c) PGI result d) PGI with geological segmentation used for the classification.
```

We can see That if we can supply helpful information to the regularization, the recovered model is more geological and closer to the ground truth.

## Geological Segmentation with Transformers

Commonly Convolutional Neural Networks are used for image segmentation but can have an inductive bias due to kernel sizes used for encoding spatial neighbours. Transformers benifit from having the notion of context derived from the entire image, not limited to kernel sizes. This notion of context allows the network to define a heatmap for every pixel of correlated cells.

```{figure} ./figures/heatmap.png
:height: 230px
:width: 595px
:name: heatmap
:alt: heatmap results
:align: center


Heat maps examples for pixels in an image ([](https://doi.org/10.48550/arXiv.2005.12872)).
```

Using the transformer we can input the geophysical model with physical parameter $\rho$. For the purpose of this study, our physical parameter is resistivity. The geophyscial model is a vector of cells $\mathbf{\Rho} = [\rho_1, \rho_2, ..., \rho_n]$. By using the SAM model [](10.48550/ARXIV.2306.11730) we can represent the mask generation as:

$$
\label{corelationmatrix}
\begin{aligned}
M_{segments} = F_{\Theta}(\mathbf{\Rho})
\end{aligned}
$$

Even not being trained on geological data, the masks from the pretrained model are quite evective at identifying structure.
- show images of the segmentations

```{figure} ./figures/geosegments.png
:height: 500px
:width: 500px
:name: segments
:alt: segment results
:align: center


Segmentations of the geophysical model.
```

```{figure} ./figures/geosegments_dip.png
:height: 500px
:width: 500px
:name: segments_dip
:alt: segment dip results
:align: center


Segmentations of the geophysical model.
```

## Geological Classification

Geological classification is done similar to the work in [Omni seg](http://arxiv.org/abs/2311.11666) that lifts 2D segmentations of a 3D object and projects the segmentation into a 3D space. Here we use simiar methods to draw geoogical structures in a 2D inversion model. 
We let $m^i \in M_{segments} (i=1,...,N_{segments})$ and to eliminate the impact of overlapping masks, create a correlation matrix $C \in \R^{N_m \times N_m}$ using the intersection of unions formula in [](#correlationmatrix)

$$
\label{corelationmatrix}
\begin{aligned}
C(i, j) = \frac{m^i \cap m^j}{m^i \cup m^j} \;\;\; \forall \;\; i,j = 1, ... , N_{segments}
\end{aligned}
$$

Which has index map $I_{maps}$. Now we will want to place bounds on the minimum amount of pixels in a mask and how many mask are to be extracted leaving us with $C_{sub} \in \R^{N_{sub} \times N_{sub}}$. From the remaing masks we want to create a hierarcial order of them. This can be done using a voting system:

$$
\label{corelationmatrix}
\begin{aligned}
v_i = \sum_{j=1}^{N_{sub}} \mathbb{I}(C_{sub}(i, j) > 0)
\end{aligned}
$$

We can then create patches $C_{ordered}$ which are ordered according to the vote counts. From here we construct a final weights matrix $\mathbf{W}$:

|       |   $W_1$ | $W_2$   | ... | $W_n$ |
| ---   |   ---   | ---     | --- |   --- |
| $W_1$ | $w_{11}$  |  $w_{12}$ | ... |$w_{1n}$ |
| $W_2$ | $w_{21}$  |  $w_{22}$ | ... |$w_{2n}$ |
| ...   |   ...   |   ...   | ... | ...   |
| $W_n$ | $w_{n1}$  |  $w_{n2}$ | ... |$w_{nn}$ |

This matrix is then used when making the geological classification. For each mask we generatte a probabiity denisty function $\mathcal{P}(\; m^i |\; W^i\;)$ in a situated mask $i$. To create the quasi-geoogical classification vector $G_{quasi}(k)$ we use the argmax function:

$$
\label{corelationmatrix}
\begin{aligned}
G_{quasi}(k) = \argmax\left[\; \mathcal{P}(\;m^n \;|\; W^n\;)\;\right] \; \in \; k = 1,.....N_{cells}
\end{aligned}
$$

Here $n$ is the deepest mask that cell $k$ is situated. Now that each cell has the properly assigned mask, we can relate back to the physical property by taking the mean value of the physical parameters situated inside a mask. Our quasi-geological model $G_{model}$ with physical with physical parameter $\rho$ has cells that are defined by the values contained inside a mask. in this instance, we will take the mean value. First we find the total number of cells inside a mask:

$$
\label{masktotal}
\begin{aligned}
K_i = \sum_{p = 1}^{N_{cells}} \mathbb{I}(m_p^i = 1) \in i = 1, ..., N_{masks}
\end{aligned}
$$

Then continue to assign the physical parameter as the mean value of the mask $m^i$:

$$
\label{corelationmatrix}
\begin{aligned}
G_{model}(k) = \frac{1}{K_{G_{quasi}(k)}}\sum_{p = 1}^{N_{cells}} \rho_p \mathbb{I}(m_p^{G_{quasi}(k)} = 1)
\end{aligned}
$$

 For example, in [](#onion_example) where the model has to travel through multiple Gaussians and we do not expect to have nested geological features. Given this prior knowledge we can set the $W_{ij}$ matrix directly with strict rules:

|       | $W_1$ | $W_2$ | $W_3$ |
| ---   | ---   | ---   | ---   |
| $W_1$ | 1     |  0    | 0     |
| $W_2$ | 0     |  1    | 0     |
| $W_3$ | 0     |  0    | 1     |
| $W_4$ | 0     |  0    | 1     |

indicating that when a cell is in $m^{i=3}$ we use the neighbourhood of $m^{i=4}$ to make the classification.

```{figure} ./figures/pgi_onion_example.png
:height: 430px
:width: 550px
:name: onion_example
:alt: geoseg results
:align: center


Recovered models and their distributions a) the true model, b) Tikhonov result, c) PGI result d) PGI with geological segmentation used for the classification.
```

## Segmentation-guided regularization

Now that we have constructed a method to influence the regularization orientation and a second that identifies structures in the geophysical model we combine the two to develop an automated framework to promote more geologically similar models. [](#framework) illustrates how the segmentation-guided regularization is integrated into the geophysical inversion. At each iteration the geophyical model is passed through the segmentation model which given prior geological information will create a weights matrix for classification. The segmented mask will then define how to regularize the objects in geophysical model by updating the regularization gradient information. Lastly, we do a petrophysical characterization, update the prior/reference model and continue to update.

```{figure} ./figures/frame-geoseg.png
:height: 450px
:width: 500px
:name: framework
:alt: geoseg framework
:align: center


Segmentation-guided regularization frame work.
```

```{figure} ./figures/geosegments_dip_bbox.png
:height: 250px
:width: 500px
:name: bbox
:alt: geoseg bound box
:align: center


Bound box example.
```

## Complex simulation

## Conclusions