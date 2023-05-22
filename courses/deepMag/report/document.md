---
# Math frontmatter:
math:
  '\magq' : '\mathbf{Q}'
  '\p'    : '\mathbf{P}'
  '\pfx'  : '\mathbf{P}_{x}'
  '\pfy'  : '\mathbf{P}_{y}'
  '\pfz'  : '\mathbf{P}_{z}'
  '\mod': '\mathbf{I}'
---

# DeepMagnetics

## Abstract

Based on the success of the graphical processing unit (GPU) in machine learning, this performance can be applied to linear problems in geophysics. In particular, the magnetics simulation can be formulated to target the GPU. With this large scaled simulations can be computed in a practical amount of time. In the following document the FFT formulated kernel with modest GPU a 1024 X 1024 receiver coverage can calculate the forward simulation in seconds. Traditional kernels would take a lot of resources and need to be highly distributed to accomplish these speeds. Starting from an already fast kernel, the fully distributed system has potential of highly parallel and fast simulated results.

## FFT's & Magnetics

Based on the work of [Jianke Qiang et. al, 2019](https://doi.org/10.1016/j.jappgeo.2019.04.009) which forms the magnetic anomaly calculation as a convolution a forward modeling kernel can be written for magnetics utilizing GPU resources allowing for fast computation of large scaled surveys. This is accomplished by utilising Fast Fourier Transforms and matrix multiplications which are highly optimized for GPU operation. In order to do the Fourier calculation, a few pieces of information must be calculated in the spatial domain. First we calculate the geometric term of the magnetic response from a single layer call it $\p_k$ can be broken into its components $\pfx$, $\pfy$, $\pfz$ represented as:

$$
\label{magPx}
\begin{aligned}
\scriptstyle \mathbf{P}_x = \frac{2X^2 - Y^2 - Z^2}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \cos(I) \sin(A) + \frac{3XY}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \cos(I) \cos(A) + \frac{3XZ}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \sin(I)
\end{aligned}
$$

$$
\label{magPy}
\begin{aligned}
\scriptstyle \mathbf{P}_y = \frac{3XY}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \cos(A) \sin(A) + \frac{2Y^2 - X^2 - Z^2}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \cos(I) \cos(A) + \frac{3YZ}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \sin(I)
\end{aligned}
$$

$$
\label{magPz}
\begin{aligned}
\scriptstyle \mathbf{P}_z = \frac{3XZ}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \cos(I) \sin(A) + \frac{3ZY}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \cos(I) \cos(A) + \frac{2Z^2 - X^2 - Y^2}{\left[ X^2 + Y^2 + Z^2 \right]^{\frac{5}{2}}} \sin(I)
\end{aligned}
$$

In [](#magPx), [](#magPy), [](#magPz) variables **X**, **Y** are defined as matrices of all the cell locations. **Z** is the depth location to the single layer being calculated. *I* is the inclination and *A* is declination of the inducing source field. Combining these, the total component contribution from a single layer is represented by:

$$
\label{magP}
\begin{aligned}
\mathbf{P}_k = \mathbf{P}_x \cos(I_0) \cos(A_0) + \mathbf{P}_y \cos(I_0) \cos(A_0) + \mathbf{P}_z \sin(I_0)
\end{aligned}
$$

Matrix [](#magP) is then used to calculate the magnetic response from a single layer. The magnetization of the layer is extracted from the model as $M_k(i, j)$ where *k* is the layer number at depth $Z_k$ of the model. The solution is then represented as:

$$
\label{qk}
\begin{aligned}
Q_k = C \cdot \p_k M_k
\end{aligned}
$$

Here $\p_k$ is a <wiki:Block_matrix> with entries that are each a <wiki:Toeplitz_matrix>. Due to the unique symetry of the Toeplitz matrix each entry can be replaced by a vector. This then allows us to calculate [](#qk) in the frequency domain transforming the matrix cross-multiplication into matrix point-multiplication [](https://doi.org/10.1016/j.jappgeo.2019.04.009):

$$
\label{qkfft}
\begin{aligned}
\tilde{\magq}'_k = \tilde{\p}_k \tilde{M}_k
\end{aligned}
$$

To reduce computation time further, the current algorithm can target the GPU by using packages such as [PyTorch](https://doi.org/10.48550/arXiv.1912.01703). GPUs are extremely optimized for mathematical operations. In particular, Fourier transforms and matrix multiplications.  

In code calculate [](#magP) for a single layer $k$ with [](#magPx), [](#magPy), [](#magPz) in the following:

```python
def psfLayer(self, Z):
    """

        :param Z: the susceptibilites of model layer Z
        :type Z: Tensor

        Note:
        I is the magnetization dip angle 
        A is the magnetization deflection angle
        I0 is the geomagnetic dip angle
        A0 is the geomagnetic deflection angle

    """

    dim2 = torch.div(self.dim,2,rounding_mode='floor')
    Dx   = self.h[0]
    Dy   = self.h[1]
    I    = self.dirs[0]
    A    = self.dirs[1]
    I0   = self.dirs[2]
    A0   = self.dirs[3]

    x   = Dx*torch.arange(-dim2[0]+1,dim2[0]+1, device=self.device)
    y   = Dy*torch.arange(-dim2[1]+1,dim2[1]+1, device=self.device)
    X,Y = torch.meshgrid(x,y)

    # Get center ready for fftshift.
    center = [1 - int(dim2[0]), 1 - int(dim2[1])]

    Rf   = torch.sqrt(X**2 + Y**2 + Z**2)**5
    PSFx = (2*X**2 - Y**2 - Z**2)/Rf*torch.cos(I)*torch.sin(A) + \
           3*X*Y/Rf*torch.cos(I)*torch.cos(A) + \
           3*X*Z/Rf*torch.sin(I)

    PSFy = 3*X*Y/Rf*torch.cos(I)*torch.sin(A) + \
           (2*Y**2 - X**2 - Z**2)/Rf*torch.cos(I)*torch.cos(A) + \
           3*Y*Z/Rf*torch.sin(I)

    PSFz = 3*X*Z/Rf*torch.cos(I)*torch.sin(A) + \
           3*Z*Y/Rf*torch.cos(I)*torch.cos(A) +\
           (2*Z**2 - X**2 - Y**2)/Rf*torch.sin(I)

    PSF  = PSFx*torch.cos(I0)*torch.cos(A0) + \
          PSFy*torch.cos(I0)*torch.sin(A0) + \
          PSFz*torch.sin(I0) 

    return PSF, center, Rf
```

The output from the above code produces matrix $\p_k$ and for an entire layer $k$ of the model subsurface. Each entry in this matrix is the summation of the each cell in that layer for each observation point on the surface. Mentioned earlier, the magnetization data from a single layer [](#qk) of the model can be calculated in the Fourier domain [](#qkfft). The calculation is then reduced to:

$$
\label{qkifft}
\begin{aligned}
\magq_k = C \cdot \magq'_k = C \cdot IFFT(\tilde{\magq}'_k)
\end{aligned}
$$

Where $C = \frac{\mu_0 \Delta x \Delta y \Delta z}{4\pi}$.

When performing the frequency domain multiplication the data are shifted in the original domain to ensure the zero frequency is in the center. This is done by applying the common operator `fftshift` found in most scientific packages (<wiki:Discrete_Fourier_transform>). This ensures that the mathematical operations in the frequency domain are applied appropriately returning the equivalent of the the original matrix cross-multiplication. `fftshift` essentially swaps the quadrants seen in [](#fftshift) transforming to zero frequency centered data in the Fourier domain.

```{figure} ./figures/fftshift.png
:height: 400px
:width: 350px
:name: fftshift
:alt: Example swapping quadrants with fftshift 
:align: center

Example swapping quadrants with fftshift ([Stephen Gruppetta, 2021](https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/)).
```
Pulling alltogether the peices outlined above, to calculate the total magnetic response, the following steps are required:
1. `fftshift` the data in spatial domain.
2. Take [](#magP) and compute $FFT(\p_k)$.
3. `fftshift` in the Fourier domain to swap data positions.
4. `fftshift` the magnetization data $\mod$.
5. $FFT(\mod)$.
6. `fftshift` to swap data positions of $\tilde{\mod}$.
7. Compute the matrix multiplication $\tilde{\p} \cdot \tilde{\mod}$
8. `fftshift` the product back the original quadrants.
9. $IFFT(\tilde{\magq}_k )$ the product of the matrix mulitplication.
10. Calculate [](#qkifft).

This is done for every layer in the model and each layer is summed to produce a 2D representation of the subsurface response. The total magnetic anomaly $\magq_k$ is then calculated by multiplying the data by constant $C$

The complete forward kernal for $\magq$ is a simple `for` loop:

```python

def forward(self, M):
    """

        Solve the forward problem using FFT

        :param M: model
        :type M: Tensor

    """

    # define the constants
    Dz = self.h[2]
    Z  = Dz/2
    dV = torch.prod(self.h)
    zeta = mu_0 / (4 * np.pi)
    Data = 0 

    # loop through each layer of the model
    for i in range(M.shape[-1]):

        # pull out the layer from the model
        I = M[:,:,i]

        # calculate the response the layer of the model
        P, center, Rf = self.psfLayer(Z)

        # use centers and the response and shift the data for FD operations
        S = torch.fft.fftshift(torch.roll(P, shifts=center, dims=[0,1]))

        # take the fft
        S = torch.fft.fft2(S)

        # shift again to swap quadrants
        S = torch.fft.fftshift(S)

        # do the same to model tensor
        I_fft = torch.fft.fftshift(I)
        I_fft = torch.fft.fft2(I_fft)
        I_fft = torch.fft.fftshift(I_fft)

        # perform the FD operations
        B = torch.fft.fftshift(S * I_fft)

        # convert back to spatial domain
        B = torch.real(torch.fft.ifft2(B))

        # add the data response from the layer
        Data = Data+B
        Z = Z + Dz

    return Data*zeta*dV
```

Even with the single python `for` loop, the algorithm can calculate a 536,870,912 cell model at 1,048,576 stations in 4.7 seconds targeting a RTX 3070 8GB GPU. This is impressively fast compare to numerous hours it could take traditional methods ([](https://doi.org/10.1016/j.jappgeo.2019.04.009)). With a distibuted system the time can be further reduced by sending each iteration of the `for` loop to an independent node with a capable GPU(s).

Furthermore, the kernel itself could be optimized further. By utilising the geometry of the receivers, there is potential to use the symetry to reduce the number of receivers required for calculation. Anther optimization but a less impactful one are to make use of pre-calculating the trignometric functions and storing them for subsequent use. Trignometric function are often costly, so reducing the amount of times to calculate common results improves the kernel's performance.

## Buried block simulation

Using the above kernel, the forward data of a magnetic block in a half space is calculated with mesh parameters:
- number of x cells = 1024
- number of y cells = 1024
- number of z cells = 512
- width of x cells  = 100 m
- width of y cell   = 100 m
- width of z cell   = 100 m

```{figure} ./figures/02-simplemodel.png
:name: simple3d
:alt: Image of a buried block model
:align: center

Buried block model.
```

Note that the axis are cell number. [Matplotlib](10.1109/MCSE.2007.55)’s `imshow()` was used for for simplicity. The forward simulations assumes a magnetization inclination and dip that are $90^o$ and along with a vertical inducing field.

```{figure} ./figures/03-simplesimulated9090.png
:name: simpledata
:alt: Simulated data using the FFT kernal
:align: center

Simulated data from the buried block model.
```

Lastly, a simulation with the centered block but this time with an inclination and declination of 45 degrees is forward modeled.

```{figure} ./figures/12-simplesimulated4545.png
:name: simpledata45
:alt: i45d45 Simulated data using the FFT kernal
:align: center

Simulated data from the buried block model with an inducing field with declination and inclination of $45^o$.
```

To compare the output of the kernel, the python package [Choclo](https://doi.org/10.5281/zenodo.7851747) was used to calculate the forward response of the same buried block. Here we see that the magnitudes are correct but the field tends to decay faster mobing away from the sphere. The Choclo formulation calculates the response of a prism while the FFT based kernel is the entire subsurface. They should be close though but differences do arise. however, with the small values of anomalous magnetic field the differences are likely caused by floating point errors.

```{figure} ./figures/11-choclo+fft.png
:name: fftcompare
:alt: FFT kernel comparison with Choclo
:align: center

FFT kernel comparison with Choclo.
```

## Complex structure simulations

To truly test the FFT based forward kernel more complex models are required. Fortunately [noddyverse](10.5194/essd-14-381-2022) has generated numerous models specifically designed for potential fields simulations ([](#noddy3d)). Moving on from simple models, large scale structural and intrusive models. With calculations done in the frequency domain objects near the edges can have harmful effects by being repeated in different quadrants ([](https://doi.org/10.1190/tle41070454.1)). However, with the shift operations during the calculation, padding can be added that removes this effect.

```{figure} ./figures/01-noddymodel3d.png
:name: noddy3d
:alt: Image of a noddy 3D model
:align: center

Example of a 3D noddyVerse model
```

Noddy models contain mostly large structural geologic structures such as folds, faults, unconformity, dykes, intrusions. These models are ideally the scale of concern for the proposed algorthm. [](#noddy2d1) is an example of the complex structures within the collection of models. For each of these models the forward simulation can be computed. [](#noddysim1) shows the result of the model from [](#noddy2d1) with an assumed vertical field.

```{figure} ./figures/05-complex7noddy.png
:name: noddy2d1
:alt: Image of a noddy 3D model
:align: center

Example of a 3D noddyVerse model
```

```{figure} ./figures/04-complex7simulated9090.png
:height: 480px
:width: 550px
:name: noddysim1
:alt: Image of a noddy 3D model
:align: center

Example of a 3D noddyVerse model
```

## Solving with conjugate gradient method

Now with a forward simulation kernal complete, we will want to see if we can minimize the difference between predicted data and observed synthetic data using the <wiki:Conjugate_gradient_method> (CG). The data difference is simply defined as the $d_{predicted} - d_{observed}$. The CG algorithm aims to minimize the objective function to find a model that fits the data. Ideally the recovered model being the one the data was generated with. The following code uses the forward kernal to solve a linear system resulting in a best fitting model thats fits the synthetic data:

```python
class CGLS(nn.Module):
    def __init__(self, forOp, CGLSit=100, eps = 1e-2, device='cuda'):
        super(CGLS, self).__init__()
        self.forOp = forOp
        self.nCGLSiter = CGLSit
        self.eps = eps

    def forward(self, b, xref):

        x = xref
        
        r = b - self.forOp(x)
        if r.norm()/b.norm()<self.eps:
                return x, r
        s = self.forOp.adjoint(r)
        
        # Initialize
        p      = s
        norms0 = torch.norm(s)
        gamma  = norms0**2

        misfit = []

        for k in range(self.nCGLSiter):
    
            q = self.forOp(p) 
            delta = torch.norm(q)**2
            alpha = gamma / delta
    
            x     = x + alpha*p
            r     = r - alpha*q

            print(k, r.norm().item()/b.norm().item())
            misfit.append(r.norm().item()/b.norm().item())
            if r.norm()/b.norm()<self.eps:
                return x, r, misfit
       
            s = self.forOp.adjoint(r)
        
            norms  = torch.norm(s)
            gamma1 = gamma
            gamma  = norms**2
            beta   = gamma / gamma1
            p      = s + beta*p
     
        return x, r, misfit

```

For simplicity, the above code is run on a similar model as [](#simple3d) but with fewer parameters:
- number of x cells = 128
- number of y cells = 128
- number of z cells = 64

[](#cg) and [](#cgsolution) display the results of running the GPU enabled algorithm. The algorithm achieves convergence and has expected decay behaviour. The recovered model is coherent with acceptable xy lateral resolution of the target but lacks any depth imaging. This is expected. Potential fields suffer from ambiguity in depth information or the lack there of. A regularization term is typically used to promote models with structure to depth ([](https://doi.org/10.1016/j.cageo.2015.09.015)).

```{figure} ./figures/09-cgiterations.png
:name: cg
:alt: Image of a CG iterations
:align: center

Conjugate gradient solving linear system convergence.
```

```{figure} ./figures/10-cgsolvemodel.png
:name: cgsolution
:alt: Image of a CG solution model
:align: center

Conjugate gradient recovered model **a)** plan view of the top surface of the model. **b)** depth slice at y = 64 m.
```

Targeting the GPU the CG iterations take on average 0.3194 seconds for a total of 0.69 seconds. Note that in each CG iteration the forward operator is executed. For large scale surveys and traditional forward kernals, each iteration is costly.

## Discussion

GPU's have been highly optimized for linear math operations brought on in the wake of the successes in areas of machine learning. The same can be applied to the magnetics geophysical forward simulation. This allows modelling capabilities for large scale airborne type surveys that can contain 10's to 100's of millions of cells to fully discretize the acquisition. What would noramlly take hours to days to calculate can be done in the fraction of the time and scalable. This means adding more resources (e.g GPU's, distributed nodes, etc.) can reduce the time until the internal communications bandwidth limits the proceedure ([](https://doi.org/10.48550/arXiv.1103.3225)).