---
# Math frontmatter:
math:
  '\magq' : '\mathbf{Q}'
  '\p'    : '\mathbf{P}'
  '\pfx'  : '\mathbf{P}_{x}'
  '\pfy'  : '\mathbf{P}_{y}'
  '\pfz'  : '\mathbf{P}_{z}'
  '\incli': '\mathbf{I}'
---

# DeepMagnetics

## FFT's & Magnetics

Based on [Jianke Qiang et. al, 2019](https://doi.org/10.1016/j.jappgeo.2019.04.009) which forms the magnetic anomaly calculation as a convolution a forward modeling kernel can be written for magnetics utilizing GPU resources allowing for fast computation of large scaled surveys. This is accomplished by utilising Fast Fourier Transforms and matrix multiplications which are highly optimized for GPU operation. In order to do the Fourier calculation, a few pieces of information must be calculated in the spatial domain. First we calculate the geometric term of the magnetic response from a single layer call it $\p$ can be broken into its components $\pfx$, $\pfy$, $\pfz$ represented as:

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

In [](#magPx), [](#magPy), [](#magPz) variables **X**, **Y** are defined as matrices of all the cell locations. **Z** is the depth location to the single layer being calculated. *I* is the inclination and *A* is declination. Combining these, the total component contribution from a single layer is represented by:

$$
\label{magP}
\begin{aligned}
\mathbf{P} = \mathbf{P}_x \cos(I_0) \cos(A_0) + \mathbf{P}_y \cos(I_0) \cos(A_0) + \mathbf{P}_z \sin(I_0)
\end{aligned}
$$

From this, the magnetization data from a single layer of the model can be coupled in the Fourier domain. To calculate the total magnetic response, it is done in following steps:
1. FFT shift the data in spatial domain.
2. Take [](#magP) and compute $FFT(\p)$.
3. FFT shift in the Fourier domain to swap Quadrants [1](https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/).
4. FFT shift the magnetization data $\incli$.
5. $FFT(\incli)$.
6. FFT shift to swap quadrants.
7. Compute the matrix multiplication $\tilde{\p} \cdot \tilde{\incli}$
8. IFFT shift the components back the original quadrants.
9. IFFT the product of the matrix mulitplication.

This is done for every layer in the model and each layer is summed to produce a 2D representation of the response from the subsurface. The total The total magnetic anomaly $\magq$ is then calculated by multiplying the data by constant $C = \frac{\mu_0 \Delta x \Delta y \Delta z}{4\pi}$

In code the geometric term calculation of a single layer is represented in the following:

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

    x = Dx*torch.arange(-dim2[0]+1,dim2[0]+1, device=self.device)
    y = Dy*torch.arange(-dim2[1]+1,dim2[1]+1, device=self.device)
    X,Y = torch.meshgrid(x,y)

    # Get center ready for fftshift.
    center = [1 - int(dim2[0]), 1 - int(dim2[1])]

    Rf = torch.sqrt(X**2 + Y**2 + Z**2)**5
    PSFx = (2*X**2 - Y**2 - Z**2)/Rf*torch.cos(I)*torch.sin(A) + \
           3*X*Y/Rf*torch.cos(I)*torch.cos(A) + \
           3*X*Z/Rf*torch.sin(I)

    PSFy = 3*X*Y/Rf*torch.cos(I)*torch.sin(A) + \
           (2*Y**2 - X**2 - Z**2)/Rf*torch.cos(I)*torch.cos(A) + \
           3*Y*Z/Rf*torch.sin(I)

    PSFz = 3*X*Z/Rf*torch.cos(I)*torch.sin(A) + \
           3*Z*Y/Rf*torch.cos(I)*torch.cos(A) +\
           (2*Z**2 - X**2 - Y**2)/Rf*torch.sin(I)

    PSF = PSFx*torch.cos(I0)*torch.cos(A0) + \
          PSFy*torch.cos(I0)*torch.sin(A0) + \
          PSFz*torch.sin(I0) 

    return PSF, center, Rf
```

The complete forward kernal for $\magq$ is as follows:

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

Note that the axis are cell number. Matplotlib’s imshow() was used her for simplicity.

The first forward simulation employed a vertical inducing field.

```{figure} ./figures/03-simplesimulated9090.png
:name: simpledata
:alt: Simulated data using the FFT kernal
:align: center

Simulated data from the buried block model.
```

The second simulation, again with a vertical inducing field but an offset block

Lastly, a simulation with the centered block but this time with an inclination and declination of 45 degrees was forward modeled.

To compare the output of the kernal, the python package [Choclo](https://doi.org/10.5281/zenodo.7851747) was used to calculate the forward response of the same buried block.

## Complex structure simulations

To truly test the FFT based forward kernal more complex models are required. Fortunately [noddyverse](10.5194/essd-14-381-2022) as generated numerous models specifically designed for potential fields simulations. Moving on from simple models, large scale structural and intrusive models.

```{figure} ./figures/01-noddymodel3d.png
:name: noddy3d
:alt: Image of a noddy 3D model
:align: center

3D noddyVerse model
```

## Discussion

[](#noddy3d)