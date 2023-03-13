import torch
import numpy as np


class IdentityMap:
    r"""Identity mapping and the base mapping class for all other SimPEG mappings.

    The ``IdentityMap`` class is used to define the mapping when
    the model parameters are the same as the parameters used in the forward
    simulation. For a discrete set of model parameters :math:`\mathbf{m}`,
    the mapping :math:`\mathbf{u}(\mathbf{m})` is equivalent to applying
    the identity matrix; i.e.:

    .. math::

        \mathbf{u}(\mathbf{m}) = \mathbf{Im}

    The ``IdentityMap`` also acts as the base class for all other SimPEG mapping classes.

    Using the *mesh* or *nP* input arguments, the dimensions of the corresponding
    mapping operator can be permanently set; i.e. (*mesh.nC*, *mesh.nC*) or (*nP*, *nP*).
    However if both input arguments *mesh* and *nP* are ``None``, the shape of
    mapping operator is arbitrary and can act on any vector; i.e. has shape (``*``, ``*``).

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        if nP is not None:
            if isinstance(nP, str):
                assert nP == "*", "nP must be an integer or '*', not {}".format(nP)
            assert isinstance(
                nP, (int, np.integer)
            ), "Number of parameters must be an integer. Not `{}`.".format(type(nP))
            nP = int(nP)
        elif mesh is not None:
            nP = mesh.nC
        else:
            nP = "*"
        self.mesh = mesh
        self._nP = nP

        super().__init__(**kwargs)

    
    def nP(self):
        r"""Number of parameters the mapping acts on.

        Returns
        -------
        int or ``*``
            Number of parameters that the mapping acts on. Returns an
            ``int`` if the dimensions of the mapping are set. If the
            mapping can act on a vector of any length, ``*`` is returned.
        """
        if self._nP != "*":
            return int(self._nP)
        if self.mesh is None:
            return "*"
        return int(self.mesh.nC)

    def shape(self):
        r"""Dimensions of the mapping operator

        The dimensions of the mesh depend on the input arguments used
        during instantiation. If *mesh* is used to define the
        identity map, the shape of mapping operator is (*mesh.nC*, *mesh.nC*).
        If *nP* is used to define the identity map, the mapping operator
        has dimensions (*nP*, *nP*). However if both *mesh* and *nP* are
        used to define the identity map, the mapping will have shape
        (*mesh.nC*, *nP*)! And if *mesh* and *nP* were ``None`` when
        instantiating, the mapping has dimensions (``*``, ``*``) and may
        act on a vector of any length.

        Returns
        -------
        tuple
            Dimensions of the mapping operator. If the dimensions of
            the mapping are set, the return is a tuple (``int``,``int``).
            If the mapping can act on a vector of arbitrary length, the
            return is a tuple (``*``, ``*``).
        """
        if self.mesh is None:
            return (self.nP, self.nP)
        return (self.mesh.nC, self.nP)

    def _transform(self, m):
        """
        Changes the model into the physical property.

        .. note::

            This can be called by the __mul__ property against a
            :meth:numpy.ndarray.

        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: transformed model

        """
        return m

    def inverse(self, D):
        """
        The transform inverse is not implemented.
        """
        raise NotImplementedError("The transform inverse is not implemented.")

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix or numpy.ndarray
            Derivative of the mapping with respect to the model parameters. For an
            identity mapping, this is just a sparse identity matrix. If the input
            argument *v* is not ``None``, the method returns the derivative times
            the vector *v*; which in this case is just *v*.

        Notes
        -----
        Let :math:`\mathbf{m}` be a set of model parameters and let :math:`\mathbf{I}`
        denote the identity map. Where the identity mapping acting on the model parameters
        can be expressed as:

        .. math::
            \mathbf{u} = \mathbf{I m},

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{I}

        For the Identity map **deriv** simply returns a sparse identity matrix.
        """
        if v is not None:
        
            return v
        
        else:
            
            return torch.sparse.spdiags(torch.ones(self.nP, 1))

    
    def dot(self, map1):
        r"""Multiply two mappings to create a :class:`SimPEG.maps.ComboMap`.

        Let :math:`\mathbf{f}_1` and :math:`\mathbf{f}_2` represent two mapping functions.
        Where :math:`\mathbf{m}` represents a set of input model parameters,
        the ``dot`` method is used to create a combination mapping:

        .. math::
            u(\mathbf{m}) = f_2(f_1(\mathbf{m}))

        Where :math:`\mathbf{f_1} : M \rightarrow K_1` and acts on the
        model first, and :math:`\mathbf{f_2} : K_1 \rightarrow K_2`, the combination
        mapping :math:`\mathbf{u} : M \rightarrow K_2`.

        When using the **dot** method, the input argument *map1* represents the first
        mapping that is be applied and *self* represents the second mapping
        that is be applied. Therefore, the correct syntax for using this method is::

            self.dot(map1)


        Parameters
        ----------
        map1 :
            A SimPEG mapping object.

        Examples
        --------
        Here we create a combination mapping that 1) projects a single scalar to
        a vector space of length 5, then takes the natural exponent.

        >>> import numpy as np
        >>> from SimPEG.maps import ExpMap, Projection

        >>> nP1 = 1
        >>> nP2 = 5
        >>> ind = np.zeros(nP1, dtype=int)

        >>> projection_map = Projection(nP1, ind)
        >>> projection_map.shape
        (5, 1)

        >>> exp_map = ExpMap(nP=5)
        >>> exp_map.shape
        (5, 5)

        >>> combo_map = exp_map.dot(projection_map)
        >>> combo_map.shape
        (5, 1)

        >>> m = np.array([2])
        >>> combo_map * m
        array([7.3890561, 7.3890561, 7.3890561, 7.3890561, 7.3890561])

        """
        return self.__mul__(map1)

    def __matmul__(self, map1):
        return self.__mul__(map1)

    __numpy_ufunc__ = True

    def __add__(self, map1):
        return SumMap([self, map1])  # error-checking done inside of the SumMap

    def __str__(self):
        return "{0!s}({1!s},{2!s})".format(
            self.__class__.__name__, self.shape[0], self.shape[1]
        )

    def __len__(self):
        return 1

    def mesh(self):
        """
        The mesh used for the mapping

        Returns
        -------
        discretize.base.BaseMesh or None
        """
        
        return self._mesh

