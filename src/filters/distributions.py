# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Defines the different distributions used in this framework
"""

from __future__ import annotations
from typing import Dict
import numpy
import scipy
from . import abstract
from .. import utils


print("-   Loading File 'distributions.py'")


class NormalDistribution(abstract.AbstractDistribution):
    r"""
    The standard multivariate normal distribution characterised by a :math:`n` dimensional mean 
    :math:`\mu` and a :math:`(n \times n)` dimensional covariance matrix :math:`\Sigma` with the 
    probability density function at :math:`x`

    .. math::
        \mathcal{N}(x|\mu, \Sigma) = \frac{1}{\sqrt{(2 \pi)^n \det \Sigma}}
               \exp\left( -\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu) \right),
    
    """
    
    @staticmethod
    def create_params(mu, P) -> Dict[str, numpy.ndarray]:
        """
        Create the parameters necessary for a multivariate normal distribution.

        :param mu: The mean(s), of shape (k,n), (k,n,1), (n,1) or (n,). That is, if multiple
                means are present, each of dimension n, then the index over the k different
                means has to come first.
        :param P: The covariance matrices of shape (k,n,n) or (n,n)
        :return: a dictionary with the necessary parameters: "mu": means, (k,n,1) shaped; "P": 
                covariance matrices, (k,n,n) shaped; "cho": lower triangular cholesky factors, 
                (k,n,n) shaped; "fac": normative factors (2*pi)**(-n/2) * det(P)**(-1/2), (k,1) 
                shaped.
        
        """
        mu = numpy.atleast_1d(numpy.squeeze(mu))
        mu = mu[:,:,None] if mu.ndim == 2 else mu[None,:,None]
        k, n = mu.shape[:2]
        
        P = numpy.atleast_2d(numpy.squeeze(P))
        assert P.ndim in [2,3], f"P's number of dimensions should be 2 or 3 (matrix or tensor), but has {P.ndim}."
        if k == 1:
            assert (P.shape[-2:] == (n,n)), (
                f"'P' should have shape (1, {n}, {n}) or ({n}, {n}) but has {P.shape}")
            P = P[None, :, :] if P.ndim == 2 else P
        else:
            assert (P.shape == (k,n,n)), (
                f"'P' should have shape ({k}, {n}, {n}) but has {P.shape}")
        
        # cho[i] @ cho[i].T = P[i], where cho[i,:,:] are lower triangular matrices
        # to multiply batchwise use numpy.einsum("Tik,Tjk->Tij", cho, cho)
        try:
            cho = numpy.linalg.cholesky(P)
        except numpy.linalg.LinAlgError as e:
            if False: # That is usually the problem
                print("Matrix not invertible. Matrix in question:\n", utils.nd_to_str(numpy.squeeze(P)))
                print("with eigenvalues:\n", utils.nd_to_str(numpy.linalg.eig(numpy.squeeze(P))[0]))
            raise e
        #print(f"cholesky test: is ", numpy.sum(P - numpy.einsum("Tik,Tjk->Tij", cho, cho)), " = 0?")
        
        # The normative factor (2*pi)**(-n/2) * det(P)**(-1/2)
        fac = (2*numpy.math.pi)**(-n/2) / numpy.prod(numpy.diagonal(cho, axis1=1, axis2=2), axis=1)
        fac = fac[:,None]
        #print("and 'P' has determinant ", numpy.prod(numpy.diagonal(cho, axis1=1, axis2=2), axis=1),
        #      "^2 which leads to a normative factor of ", fac[:,0])
        return {"mu":mu, "P":P, "cho":cho, "fac":fac}

    def __init__(self, seed: int | numpy.random.Generator | None = None, peek: bool = False,
                       params: Dict[str, numpy.ndarray] = None, **kwargs) -> None:
        """
        Creates a multivariate normal distribution with mean ``mu`` and covariance ``covar``.

        :param mu: The mean(s), of shape (k,n), (k,n,1), (n,1) or (n,). That is, if multiple
                means are present, each of dimension n, then the index over the k different
                means has to come first.
        :param P: The covariance matrices of shape (k,n,n) or (n,n)
        
        """
        if params is not None:
            n = params["mu"].shape[-2]
            params["mu"] = numpy.atleast_1d(numpy.squeeze(params["mu"])).reshape((-1,n,1))
            k = params["mu"].shape[0]
            params["P"] = numpy.atleast_2d(numpy.squeeze(params["P"])).reshape((k,n,n))
            params["cho"] = numpy.atleast_2d(numpy.squeeze(params["cho"])).reshape((k,n,n))
            params["fac"] = numpy.atleast_2d(numpy.squeeze(params["fac"])).reshape((k,1))
        super().__init__(seed, peek, params, **kwargs)
        self.k, self.n = self._params["mu"].shape[:2]
    
    def marginal(self, comp):
        r"""
        Returns the marginal distribution of the given components.

        For example if we have a multivariate normal of 4 components 

        .. math::
            \mathcal{N}\left(
                \left( \begin{array}{c} \!\mu_0\\ \!\mu_1\\ \!\mu_2\\ \!\mu_3\end{array} \right), 
                \left( \begin{array}{cccc} 
                    \!\Sigma_{00}& \Sigma_{01}& \Sigma_{02}& \Sigma_{03}\\
                    \!\Sigma_{01}& \Sigma_{11}& \Sigma_{12}& \Sigma_{13}\\
                    \!\Sigma_{02}& \Sigma_{12}& \Sigma_{22}& \Sigma_{23}\\
                    \!\Sigma_{03}& \Sigma_{13}& \Sigma_{23}& \Sigma_{33}\\
                \end{array} \right)
            \right) 
    
        then the marginal distribution of components :math:`[1,3]` is 

        .. math::
            \mathcal{N}\left( 
                \left( \begin{array}{c} \!\mu_1\\ \!\mu_3\end{array} \right), 
                \left( \begin{array}{cccc} 
                    \!\Sigma_{11}& \Sigma_{13}\\
                    \!\Sigma_{13}& \Sigma_{33}\\
                \end{array} \right)
                \right) 
    
        :param comp: The components of the marginal distribution. Needs to be sorted in ascending order.

        """
        comp = sorted(numpy.atleast_1d(comp).flatten())
        assert comp[0] >= 0 and comp[-1] < self.n, f"components {comp} out of bounds for distribution with {self.n} distributions."
        new_mu = self._params["mu"][:,comp,:]
        new_P  = self._params["P"][:,comp,:][:,:,comp]
        return NormalDistribution(seed = self._rng_gen, peek=True, mu=new_mu, P=new_P)


    def mean(self):
        """
        Returns the mean of the distribution.
        :returns: the (:math:`n`,) dimensional mean vector
        :rtype: numpy.ndarray 
        """
        return self._params["mu"]

    def cov(self):
        """
        Returns the covariance matrix of the distribution.
        :returns: the (:math:`n`,:math:`n`) dimensional covariance matrix
        :rtype: numpy.ndarray 
        """
        return self._params["P"]

    def pdf(self, x):
        """
        Gives the probability density function evaluated at :math:`m` different points :math:`x`. 

        :param x: The evaluation point of the density. If the :math:`k` different densities should 
                be evaluated at :math:`m` multiple points the input should have shape 
                (:math:`k`,:math:`n`,:math:`m`). If :math:`k=1` or :math:`m=1` these axes can be 
                left out. Alternatively, if every density should be evaluated at the same points, 
                the axes of :math:`k` can also be left out or have length one.
                Note that if :math:`k=n` a (:math:`n`,:math:`n`) input will be interpreted as 
                (:math:`n`, :math:`n`, :math:`1`) and not (:math:`1`, :math:`n`, :math:`n`). Thus it 
                is always preferred to give the input as a tensor.
        :type x: numpy.ndarray
        :returns: The density at the given points, has shape (k,m)
        :rytpe: numpy.ndarray
        """
        k, n = (self.k, self.n)
        x = numpy.atleast_1d(x)
        if x.ndim == 3:
            assert (x.shape[0] in [1,k]) and (x.shape[1] == n), (
                   f"input 'x' should have shape ({k}, {n}, m) or (1, {n}, m), but has {x.shape}")
        elif x.ndim == 2:
            if x.shape in [(k, n), (1, n)]:
                x = x[:,:,None]
            else:
                assert x.shape[0] == n, (f"x has shape {x.shape}, but since the first "+
                                              f"dimension is not n={n}, it should have shape"+
                                              f" (k,n)=({k},{n}) or (1,{n})")
                x = x[None, :, :]
        elif x.ndim == 1:
            if x.shape[0] == n:
                x = x[None,:,None]
            elif n == 1:
                if x.shape[0] == k:
                    x = x[:,None,None]
                else:
                    x = x[None,None,:]
            else:
                raise ValueError(f"if only a vector is given it should have shape ({n},) "
                  +(f"or ({k},) or (m,)" if n==1 else "")+f"but has {x.shape}")
        else:
            assert False, f"x.ndim ({x.ndim}) should be 1, 2 or 3."

        #prepare results array, will have shape (k,n,m) by broadcasting since both are already tensors
        normed = x - self._params["mu"]
        cho = self._params["cho"]
        for i in range(k):
            # since no batchwise solve_triangular method exists (at least in numpy/scipy currently),
            # use this in place scipy method k times
            scipy.linalg.solve_triangular(cho[i], normed[i], lower=True, overwrite_b=True)
        return self._params["fac"] * numpy.exp(-1/2 * (normed**2).sum(axis=1) )

    def logpdf(self, x):
        """
        Returns the logarithm of the :attr:`pdf`.

        :param x: The evaluation point of the density. Either of shape (n,) or (n,k) if the the 
                density should be evaluated at multiple (k) points.
        :type x: numpy.ndarray
        """
        k, n = (self.k, self.n)
        x = numpy.atleast_1d(x)
        if x.ndim == 3:
            assert (x.shape[0] in [1,k]) and (x.shape[1] == n), (
                   f"input 'x' should have shape ({k}, {n}, m) or (1, {n}, m), but has {x.shape}")
        elif x.ndim == 2:
            if x.shape in [(k, n), (1, n)]:
                x = x[:,:,None]
            else:
                assert x.shape[0] == n, (f"x has shape {x.shape}, but since the first "+
                                              f"dimension is not n={n}, it should have shape"+
                                              f" (k,n)=({k},{n}) or (1,{n})")
                x = x[None, :, :]
        elif x.ndim == 1:
            if x.shape[0] == n:
                x = x[None,:,None]
            elif n == 1:
                if x.shape[0] == k:
                    x = x[:,None,None]
                else:
                    x = x[None,None,:]
            else:
                raise ValueError(f"if only a vector is given it should have shape ({n},) "
                  +(f"or ({k},) or (m,)" if n==1 else "")+f"but has {x.shape}")
        else:
            assert False, f"x.ndim ({x.ndim}) should be 1, 2 or 3."
        
        #prepare results array, will have shape (k,n,m) by broadcasting since both are already tensors
        normed = x - self._params["mu"]
        cho = self._params["cho"]
        for i in range(k):
            # since no batchwise solve_triangular method exists (at least in numpy/scipy currently),
            # use this in place scipy method k times
            scipy.linalg.solve_triangular(cho[i], normed[i], lower=True, overwrite_b=True)
        return numpy.log(self._params["fac"] ) - 1/2 * (normed**2).sum(axis=1)

    def rvs(self, m=1):
        """
        Samples :math:`n` random variables according to the distribution

        :param m: Number of generated samples per distribution
        :type m: int
        :return: :math:`m` random samples of the normal distribution with shape 
                (:math:`k`,:math:`n`,:math:`m`)
        """
        standard_samples = self._rng_gen.normal(size=(self.k, self.n, m))
        result = self._params["cho"] @ standard_samples + self._params["mu"]
        return result






class StudentTDistribution(abstract.AbstractDistribution):
    r"""
    The multivariate student-t distribution characterised by a :math:`n` dimensional mean 
    :math:`\mu`, a :math:`(n \times n)` dimensional scale matrix :math:`\Sigma` and the degrees of 
    freedom :math:`\nu`, with the probability density function at :math:`x`

    .. math::
        t_{\nu}(x|\mu, \Sigma) = \frac{\Gamma(\frac{\nu+p}{2})}{\Gamma(\frac{\nu}{2}) \nu^{\frac{p}{2}} 
                                        \pi^{\frac{p}{2}} \sqrt{\det(\Sigma)} } 
                                \left[ 1 + \frac{1}{\nu}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right]^{\frac{\nu+p}{2}}
    """
    
    @staticmethod
    def create_params(mu, P, nu, P_covar:bool=False) -> Dict[str, numpy.ndarray]:
        """
        Create the parameters necessary for a multivariate normal distribution.

        :param mu: The mean(s), of shape (k,n), (k,n,1), (n,1) or (n,). That is, if multiple
                means are present, each of dimension n, then the index over the k different
                means has to come first.
        :param P: The scale matrices of shape (k,n,n) or (n,n)
        :param nu: The degrees of freedom of shape () or (k,)
        :param P_covar: Whether P is a covariance and the scale matrix has to be scaled up using
                Kullback Leibler minimisation
        :return: a dictionary with the necessary parameters: "mu": means, (k,n,1) shaped; "P": 
                covariance matrices, (k,n,n) shaped; "cho": lower triangular cholesky factors, 
                (k,n,n) shaped; "fac": normative factors (2*pi)**(-n/2) * det(P)**(-1/2), (k,1) 
                shaped.
        """
        mu = numpy.atleast_1d(numpy.squeeze(mu))
        mu = mu[:,:,None] if mu.ndim == 2 else mu[None,:,None]

        k, n = mu.shape[:2]
        
        P = numpy.atleast_2d(numpy.squeeze(P))
        assert P.ndim in [2,3], f"P's number of dimensions should be 2 or 3 (matrix or tensor), but has {P.ndim}."
        if k == 1:
            assert (P.shape[-2:] == (n,n)), (
                f"'P' should have shape (1, {n}, {n}) or ({n}, {n}) but has {P.shape}")
            P = P[None, :, :] if P.ndim == 2 else P
        else:
            assert (P.shape == (k,n,n)), (
                f"'P' should have shape ({k}, {n}, {n}) but has {P.shape}")
        nu = numpy.atleast_1d(nu).astype(float)
        assert (nu.shape in [ (k,), (1,)]), "'nu' should have shape ({k},) or (1,) or be a scalar"
        
        if P_covar:
            KLDmins = numpy.array([utils.KLDmin_Norm_T(dim=n, nu=nux) for nux in nu])[:,None,None]
            P = KLDmins * P

        # cho @ cho.T = P, where cho[i,:,:] are lower triangular matrices
        cho = numpy.linalg.cholesky(P)
        # The normative factor gamma(nu/2+p/2) / ( gamma(nu/2) / nu**p/2 / pi**p/2 det(cho))
        fac = (nu*numpy.math.pi)**(-n/2) / numpy.prod(numpy.diagonal(cho, axis1=1, axis2=2), axis=1)
        fac = scipy.special.gamma((nu+n)/2) / scipy.special.gamma(nu/2) * fac
        fac = fac[:,None]
        nu  = nu[:,None]
        return {"mu":mu, "P":P, "nu":nu, "cho":cho, "fac":fac}

    def __init__(self, seed: int | numpy.random.Generator | None = None, peek: bool = False,
                       params: Dict[str, numpy.ndarray] = None, **kwargs) -> None:
        """
        Creates a multivariate student-t distribution with mean ``mu``, covariance ``covar`` and 
        degrees of freedom ``nu``.

        :param mu: The mean(s), of shape (k,n), (k,n,1), (n,1) or (n,). That is, if multiple
                means are present, each of dimension n, then the index over the k different
                means has to come first.
        :param P: The scale matrices of shape (k,n,n) or (n,n)
        :param nu: The degrees of freedom of shape () or (k,)
        :param P_covar: Whether P is a covariance and the scale matrix has to be scaled up using
                Kullback Leibler minimisation
        """
        if params is not None:
            n = params["mu"].shape[-2]
            params["mu"] = numpy.atleast_1d(numpy.squeeze(params["mu"])).reshape((-1,n,1))
            k = params["mu"].shape[0]
            params["P"] = numpy.atleast_2d(numpy.squeeze(params["P"])).reshape((k,n,n))
            params["cho"] = numpy.atleast_2d(numpy.squeeze(params["cho"])).reshape((k,n,n))
            params["fac"] = numpy.atleast_2d(numpy.squeeze(params["fac"])).reshape((k,1))
            params["nu"] = numpy.atleast_2d(numpy.squeeze(params["nu"])).reshape((k,1))
        super().__init__(seed, peek, params, **kwargs)
        self.k, self.n = self._params["mu"].shape[:2]

    def marginal(self, comp):
        r"""
        Returns the marginal distribution of the given components.

        For example if we have a multivariate student-t distribution of 4 components 

        .. math::
            t_\nu\left(
                \left( \begin{array}{c} \!\mu_0\\ \!\mu_1\\ \!\mu_2\\ \!\mu_3\end{array} \right), 
                \left( \begin{array}{cccc} 
                    \!\Sigma_{00}& \Sigma_{01}& \Sigma_{02}& \Sigma_{03}\\
                    \!\Sigma_{01}& \Sigma_{11}& \Sigma_{12}& \Sigma_{13}\\
                    \!\Sigma_{02}& \Sigma_{12}& \Sigma_{22}& \Sigma_{23}\\
                    \!\Sigma_{03}& \Sigma_{13}& \Sigma_{23}& \Sigma_{33}\\
                \end{array} \right)
            \right) 
    
        then the marginal distribution of components :math:`[1,3]` is 

        .. math::
            t_\nu\left( 
                \left( \begin{array}{c} \!\mu_1\\ \!\mu_3\end{array} \right), 
                \left( \begin{array}{cccc} 
                    \!\Sigma_{11}& \Sigma_{13}\\
                    \!\Sigma_{13}& \Sigma_{33}\\
                \end{array} \right)
                \right) 
    
        :param comp: The components of the marginal distribution. Needs to be sorted in ascending order.

        """
        comp = sorted(numpy.atleast_1d(comp).flatten())
        assert comp[0] >= 0 and comp[-1] < self.n, f"components {comp} out of bounds for distribution with {self.n} distributions."
        new_mu = self._params["mu"][:,comp,:]
        new_P  = self._params["P"][:,comp,:][:,:,comp]
        return StudentTDistribution(seed = self._rng_gen, peek=True, mu=new_mu, P=new_P, nu=self._params["nu"][:,0])
    

    def mean(self):
        """
        Returns the mean of the distribution.
        :returns: the (:math:`n`,) dimensional mean vector
        :rtype: numpy.ndarray 
        """
        return self._params["mu"]

    def cov(self):
        """
        Returns the covariance matrix of the distribution.
        :returns: the (:math:`n`,:math:`n`) dimensional covariance matrix
        :rtype: numpy.ndarray 
        """
        factor = self._params["nu"]
        factor[factor <= 2+1e-10] = numpy.NaN
        return (factor/(factor-2))[:,:,None] *  self._params["P"]

    def pdf(self, x):
        """
        Gives the probability density function evaluated at :math:`x`.

        :param x: The evaluation point of the density. If the :math:`k` different densities should 
                be evaluated at :math:`m` multiple points the input should have shape 
                (:math:`k`,:math:`n`,:math:`m`). If :math:`k=1` or :math:`m=1` these axes can be 
                left out. Alternatively, if every density should be evaluated at the same points, 
                the axes of :math:`k` can also be left out or have length one.
                Note that if :math:`k=n` a (:math:`n`,:math:`n`) input will be interpreted as 
                (:math:`n`, :math:`n`, :math:`1`) and not (:math:`1`, :math:`n`, :math:`n`). Thus it 
                is always preferred to give the input as a tensor.
        :type x: numpy.ndarray
        :returns: The density at the given points, has shape (k,m)
        :rytpe: numpy.ndarray
        """
        k, n = (self.k, self.n)
        x = numpy.atleast_1d(x)
        if x.ndim == 3:
            assert (x.shape[0] in [1,k]) and (x.shape[1] == n), (
                   f"input 'x' should have shape ({k}, {n}, m) or (1, {n}, m), but has {x.shape}")
        elif x.ndim == 2:
            if x.shape in [(k, n), (1, n)]:
                x = x[:,:,None]
            else:
                assert x.shape[0] == n, (f"x has shape {x.shape}, but since the first "+
                                              f"dimension is not n={n}, it should have shape"+
                                              f" (k,n)=({k},{n}) or (1,{n})")
                x = x[None, :, :]
        elif x.ndim == 1:
            if x.shape[0] == n:
                x = x[None,:,None]
            elif n == 1:
                if x.shape[0] == k:
                    x = x[:,None,None]
                else:
                    x = x[None,None,:]
            else:
                raise ValueError(f"if only a vector is given it should have shape ({n},) "
                  +(f"or ({k},) or (m,)" if n==1 else "")+f"but has {x.shape}")
        else:
            assert False, f"x.ndim ({x.ndim}) should be 1, 2 or 3."
        
        #prepare results array, will have shape (k,n,m) by broadcasting since both are already tensors
        normed = x - self._params["mu"]
        cho = self._params["cho"]
        nu = self._params["nu"]
        for i in range(k):
            # since no batchwise solve_triangular method exists (at least in np/scipy currently),
            # use this in place scipy method k times
            scipy.linalg.solve_triangular(cho[i], normed[i], lower=True, overwrite_b=True)
        return self._params["fac"] * numpy.power(1 + (normed**2).sum(axis=1)/nu, -(nu+n)/2)

    def logpdf(self, x):
        """
        Returns the logarithm of the :attr:`pdf`.

        :param x: The evaluation point of the density. Either of shape (n,) or (n,k) if the the 
                density should be evaluated at multiple (k) points.
        :type x: numpy.ndarray
        """
        k, n = (self.k, self.n)
        x = numpy.atleast_1d(x)
        if x.ndim == 3:
            assert (x.shape[0] in [1,k]) and (x.shape[1] == n), (
                   f"input 'x' should have shape ({k}, ({n}, m)) or (1, ({n}, m)), but has {x.shape}")
        elif x.ndim == 2:
            if x.shape in [(k, n), (1, n)]:
                x = x[:,:,None]
            else:
                assert x.shape[0] == n, (f"x has shape {x.shape}, but since the first "+
                                              f"dimension is not n={n}, it should have shape"+
                                              f" (k,n)=({k},{n}) or (1,{n})")
                x = x[None, :, :]
        elif x.ndim == 1:
            if x.shape[0] == n:
                x = x[None,:,None]
            elif n == 1:
                if x.shape[0] == k:
                    x = x[:,None,None]
                else:
                    x = x[None,None,:]
            else:
                raise ValueError(f"if only a vector is given it should have shape ({n},) "
                  +(f"or ({k},) or (m,)" if n==1 else "")+f"but has {x.shape}")
        else:
            assert False, f"x.ndim ({x.ndim}) should be 1, 2 or 3."
        
        #prepare results array, will have shape (k,n,m) by broadcasting since both are already tensors
        normed = x - self._params["mu"]
        cho = self._params["cho"]
        nu = self._params["nu"]
        for i in range(k):
            # since no batchwise solve_triangular method exists (at least in np/scipy currently),
            # use this in place scipy method k times
            scipy.linalg.solve_triangular(cho[i], normed[i], lower=True, overwrite_b=True)
        return numpy.log(self._params["fac"] ) - (nu+n)/2 * numpy.log1p( (normed**2).sum(axis=1)/nu )

    def rvs(self, m=1):
        """
        Samples :math:`m` random variables according to the distribution

        :param m: Number of generated samples per distribution
        :type m: int
        :return: :math:`m` random samples of the normal distribution with shape 
                (:math:`k`,:math:`n`,:math:`m`), where dimensions of size :math:`1` are removed, i.e.
                when only a single distribution is represented and :math:`m` is not specified will 
                produce a (:math:`n`,) shaped array
        """
        standard_samples = self._rng_gen.normal(size=(self.k, self.n, m))
        nu = self._params["nu"]
        chi2 = self._rng_gen.chisquare(df=nu[:,0], size=(m,self.k)).T
        standard_samples = standard_samples * numpy.sqrt(nu/chi2)[:, None, :]
        result = self._params["cho"] @ standard_samples + self._params["mu"]
        return result




class Joint_Indep_Distribution(abstract.AbstractDistribution):
    r"""
    The joint distribution of two independent random variables. That is, if :math:`X` has density 
    :math:`p_X(\cdot)` and :math:`Y` has density :math:`p_Y(\cdot)` then this class represents the
    distribution of :math:`(X, Y)^\top` with joint density

    .. math::
        p_{XY}(x,y) = p_X(x) \cdot p_Y(y).
    
    """
    
    @staticmethod
    def create_params(distributions, dist_kwargs) -> Dict[str, numpy.ndarray]:
        """
        Create the parameters of the joint distribution by recursively calling underlying 
        create_params of the individual distributions

        :param list_of_distr_classes: a list of distribution classes.
        :param list_of_kwarg_dicts: a list of dictionaries, each holding the parameters to create
            each individual component - distribution
        """

        list_of_param_dicts = []
        l = len(distributions)
        for i in range(l):
            list_of_param_dicts.append(distributions[i].create_params(**dist_kwargs[i]))
        
        joined_dict = { f"{i}_"+name: param for i in range(l)
                                            for name, param in list_of_param_dicts[i].items() }
        return joined_dict
    
    @staticmethod
    def join_distr(*distributions, seed=None, peek=True):
        """
        Joins the list of distributions into a single joint distribution. The parameters are shared
        and the random number generator is induced by the first distribution if not specified further.
        """
        seed = distributions[0]._rng_gen if seed is None else seed
        return Joint_Indep_Distribution(
                seed=seed, peek=peek, 
                params={ f"{i}_"+name: param for i in range(len(distributions)) 
                                             for name, param in distributions[i]._params.items() },
                distributions=[dist.__class__ for dist in distributions],
            )

    def __init__(self, seed: int | numpy.random.Generator | None = None, peek: bool = False,
                       params: Dict[str, numpy.ndarray] = None, **kwargs) -> None:
        """
        Creates the joint distribution.

        :param list_of_distr_classes: a list of distribution classes.
        :param list_of_kwarg_dicts: a list of dictionaries, each holding the parameters to create
            each individual component - distribution
        
        """
        super().__init__(seed, peek, params, **kwargs)
        distributions = kwargs["distributions"]
        self.l = len(distributions)
        list_of_param_dicts = [{} for i in range(self.l)]
        for name, param in self._params.items():
            split = name.split("_")
            i = int(split[0])
            orig_name = "_".join(split[1:])
            list_of_param_dicts[i][orig_name] = param
        self.distributions = [distributions[i](seed=self._rng_gen, params=list_of_param_dicts[i]) 
                               for i in range(self.l)]
        self.n = sum( [dist.n for dist in self.distributions] )

    
    def marginal(self, comp):
        r"""
        Returns the marginal distribution of the given components.

        For example if we have a multivariate student-t distribution of 4 components 

        .. math::
            t_\nu\left(
                \left( \begin{array}{c} \!\mu_0\\ \!\mu_1\\ \!\mu_2\\ \!\mu_3\end{array} \right), 
                \left( \begin{array}{cccc} 
                    \!\Sigma_{00}& \Sigma_{01}& \Sigma_{02}& \Sigma_{03}\\
                    \!\Sigma_{01}& \Sigma_{11}& \Sigma_{12}& \Sigma_{13}\\
                    \!\Sigma_{02}& \Sigma_{12}& \Sigma_{22}& \Sigma_{23}\\
                    \!\Sigma_{03}& \Sigma_{13}& \Sigma_{23}& \Sigma_{33}\\
                \end{array} \right)
            \right) 
    
        then the marginal distribution of components :math:`[1,3]` is 

        .. math::
            t_\nu\left( 
                \left( \begin{array}{c} \!\mu_1\\ \!\mu_3\end{array} \right), 
                \left( \begin{array}{cccc} 
                    \!\Sigma_{11}& \Sigma_{13}\\
                    \!\Sigma_{13}& \Sigma_{33}\\
                \end{array} \right)
                \right) 
    
        :param comp: The components of the marginal distribution. Needs to be sorted in ascending order.

        """
        comp = sorted(numpy.atleast_1d(comp).flatten())
        assert comp[0] >= 0 and comp[-1] < self.n, f"components {comp} out of bounds for distribution with {self.n} distributions."

        marg_dist = []
        sumn, comp_idx = (0,0)
        for i in range(self.l):
            dist = self.distributions[i]
            lastn = sumn + dist.n
            sub_comp = []
            while comp[comp_idx] < lastn:
                sub_comp.append(comp[comp_idx]-sumn)
                comp_idx += 1
            if len(sub_comp) > 0:
                marg_dist.append(dist.marginal(sub_comp))
            sumn = lastn

        return Joint_Indep_Distribution.join_distr(marg_dist)
    

    def mean(self):
        """
        Returns the mean of the distribution.
        :returns: the (:math:`n`,) dimensional mean vector
        :rtype: numpy.ndarray 
        """
        return numpy.concatenate([dist.mean() for dist in self.distributions], axis=-2)

    def cov(self):
        """
        Returns the covariance matrix of the distribution.
        :returns: the (:math:`n`,:math:`n`) dimensional covariance matrix
        :rtype: numpy.ndarray 
        """
        cov = numpy.zeros((self.k,self.n,self.n))
        start = 0
        for i in range(self.l):
            dist = self.distributions[i]
            n = dist.n
            cov[:,start:start+n,start:start+n] = dist.cov()
            start += n
        return cov

    def pdf(self, x):
        """
        Gives the probability density function evaluated at :math:`m` different points :math:`x`. 

        :param x: The evaluation point of the density. If the :math:`k` different densities should 
                be evaluated at :math:`m` multiple points the input should have shape 
                (:math:`k`,:math:`n`,:math:`m`). If :math:`k=1` or :math:`m=1` these axes can be 
                left out. Alternatively, if every density should be evaluated at the same points, 
                the axes of :math:`k` can also be left out or have length one.
                Note that if :math:`k=n` a (:math:`n`,:math:`n`) input will be interpreted as 
                (:math:`n`, :math:`n`, :math:`1`) and not (:math:`1`, :math:`n`, :math:`n`). Thus it 
                is always preferred to give the input as a tensor.
        :type x: numpy.ndarray
        :returns: The density at the given points, has shape (k,m)
        :rytpe: numpy.ndarray
        """
        k, n = (self.k, self.n)
        x = numpy.atleast_1d(x)
        if x.ndim == 3:
            assert (x.shape[0] in [1,k]) and (x.shape[1] == n), (
                   f"input 'x' should have shape ({k}, {n}, m) or (1, {n}, m), but has {x.shape}")
        elif x.ndim == 2:
            if x.shape in [(k, n), (1, n)]:
                x = x[:,:,None]
            else:
                assert x.shape[0] == n, (f"x has shape {x.shape}, but since the first "+
                                              f"dimension is not n={n}, it should have shape"+
                                              f" (k,n)=({k},{n}) or (1,{n})")
                x = x[None, :, :]
        elif x.ndim == 1:
            if x.shape[0] == n:
                x = x[None,:,None]
            elif n == 1:
                if x.shape[0] == k:
                    x = x[:,None,None]
                else:
                    x = x[None,None,:]
            else:
                raise ValueError(f"if only a vector is given it should have shape ({n},) "
                  +(f"or ({k},) or (m,)" if n==1 else "")+f"but has {x.shape}")
        else:
            assert False, f"x.ndim ({x.ndim}) should be 1, 2 or 3."
        
        prod = None
        start = 0
        for i in range(self.l):
            dist = self.distributions[i]
            n = dist.n
            prod = dist.pdf(x[:,start:start+n,:]) if prod is None else prod * dist.pdf(x[:,start:start+n,:])
            start += n
        return prod

    def logpdf(self, x):
        """
        Returns the logarithm of the :attr:`pdf`.

        :param x: The evaluation point of the density. Either of shape (n,) or (n,k) if the the 
                density should be evaluated at multiple (k) points.
        :type x: numpy.ndarray
        """
        k, n = (self.k, self.n)
        x = numpy.atleast_1d(x)
        if x.ndim == 3:
            assert (x.shape[0] in [1,k]) and (x.shape[1] == n), (
                   f"input 'x' should have shape ({k}, {n}, m) or (1, {n}, m), but has {x.shape}")
        elif x.ndim == 2:
            if x.shape in [(k, n), (1, n)]:
                x = x[:,:,None]
            else:
                assert x.shape[0] == n, (f"x has shape {x.shape}, but since the first "+
                                              f"dimension is not n={n}, it should have shape"+
                                              f" (k,n)=({k},{n}) or (1,{n})")
                x = x[None, :, :]
        elif x.ndim == 1:
            if x.shape[0] == n:
                x = x[None,:,None]
            elif n == 1:
                if x.shape[0] == k:
                    x = x[:,None,None]
                else:
                    x = x[None,None,:]
            else:
                raise ValueError(f"if only a vector is given it should have shape ({n},) "
                  +(f"or ({k},) or (m,)" if n==1 else "")+f"but has {x.shape}")
        else:
            assert False, f"x.ndim ({x.ndim}) should be 1, 2 or 3."
        
        prod = None
        start = 0
        for i in range(self.l):
            dist = self.distributions[i]
            n = dist.n
            prod = dist.logpdf(x[:,start:start+n,:]) if prod is None else prod + dist.logpdf(x[:,start:start+n,:])
            start += n
        return prod

    def rvs(self, m=1):
        """
        Samples :math:`n` random variables according to the distribution

        :param m: Number of generated samples per distribution
        :type m: int
        :return: :math:`m` random samples of the normal distribution with shape 
                (:math:`k`,:math:`n`,:math:`m`), where dimensions :math:`m` and :math:`k` of size :math:`1` are removed, i.e.
                when only a single distribution is represented and :math:`m` is not specified will 
                produce a (:math:`n`,) shaped array
        """
        sample = numpy.zeros((self.k,self.n,m))
        start = 0
        for i in range(self.l):
            dist = self.distributions[i]
            n = dist.n
            sample[:,start:start+n,:] = dist.rvs(m=m)
            start += n
        return sample
