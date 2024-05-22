# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Some utility functions.
"""
import numpy as np
import sys
import sympy
import scipy.stats as stats
from scipy.special import gamma
from scipy.fft import ifft, fftshift, ifftshift
from scipy.linalg import solve_triangular

print("-  Loading File 'src/utils.py'")

def exp_wrt_gamma(f, alpha, beta, n=1000):
    r"""
    Calculates the expectation :math:`\mathbb{E}_{x\sim\Gamma(\alpha, \beta)}\left[f(x)\right]`.
    Here the Gamma distribution has density

    .. math:: \Gamma(x | \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1}\cdot e^{-\beta x}
    """
    # the nodes when integrating over the interval [0,1], i.e. midpoints of the intervals [i/n, (i+1)/n] for i = 0, ..., n-1
    nodes = (np.arange(n)+.5)/n
    # the nodes when integrating over a gamma distribution, chosen such that the same amount of probability mass is between each node
    gamma_nodes = stats.gamma.ppf(nodes, alpha, scale=1/beta)
    return np.sum(f(gamma_nodes), axis=0)/n

def KLD_Norm_T(dim, nu, a):
    r"""
    calculates the Kullback-Leibler divergence :math:`\text{KL}\big(\,\mathcal{N}(\mu, \Sigma) \,\|\, t_{\nu}\,(\mu, a\cdot \Sigma)\, \big)`
    depending on :math:`a`.

    :param dim: The dimension of the space of which the distributions are defined, i.e. :math:`\mu`
        is a ``dim``-dimensionaler vektor
    :type dim: int
    :param nu: The degrees of freedom of the student-t distribution
    :type nu: int, can be float
    :param a: The points at which the Kullback-Leibler divergence should be evaluated
    :type a: float or vector of floats
    :returns: The evaluated Kullback-Leibler divergences, same shape as ``a``
    """

    a = np.atleast_1d(a)
    f = lambda x: np.log(1 + x[:,None]/nu/np.atleast_1d(a)[None, :])
    return np.log( gamma(nu/2)/gamma((nu+dim)/2) * (nu/2)**(dim/2) ) + dim/2 * np.log(a) - dim/2 + (nu+dim)/2 * exp_wrt_gamma(f, alpha=dim/2, beta=1/2)

old_KLDmin = {}
def KLDmin_Norm_T(dim, nu):
    r"""
    Minimises the Kullback-Leibler divergence between a normal distribution and a student-t distribution.

    This is done by finding the minimising parameter :math:`a` in the following expression

    .. math:: 
        \mathop{\mathrm{argmin}}_{a > 0}\ \text{KL}\big(\,\mathcal{N}(\mu, \Sigma) \,\|\, t_{\nu}\,(\mu, a\cdot \Sigma)\, \big),
    
    or equivalently
    
    .. math:: 
        \mathop{\mathrm{argmin}}_{a > 0}\ \text{KL}\big(\,\mathcal{N}(\mu, \tfrac{1}{a}\Sigma) \,\|\, t_{\nu}\,(\mu, \Sigma)\, \big),
    
    
    which is independent of :math:`\mu` and :math:`\Sigma` since in the Kullback-Leibler divergence

    .. math:: 
        \text{KL}\big(\,\mathcal{N}(\mu, \Sigma) \,\|\, t_{\nu}\,(\mu, a\cdot \Sigma)\, \big)
        = \int_{\mathbb{R}^{d}} \mathcal{N}(x | \mu, \Sigma) \cdot \log\left(\frac{
            \mathcal{N}(x | \mu, \Sigma) }{ t_{\nu}\,(x | \mu, a\cdot \Sigma) }\right) \ \text{d}x
    
    we can apply a transformation :math:`x = \Sigma^{\frac{1}{2}}y +\mu`
    
    :param dim: The dimension of the space of which the distributions are defined, i.e. :math:`\mu`
        is a ``dim``-dimensionaler vektor
    :type dim: int
    :param nu: The degrees of freedom of the student-t distribution
    :type nu: int, can be float
    :returns: the minimising parameter :math:`a > 0`
    """ 
    if (dim,nu) in old_KLDmin.keys():
        return old_KLDmin[(dim,nu)]
    a, a_old = (1., 0)
    while np.abs(a-a_old)/np.abs(a) > 1e-5:
        a_old = a
        uG = float(       sympy.functions.special.gamma_functions.uppergamma(1-dim/2, nu*a/2) if dim != 2 
                    else -sympy.functions.special.error_functions.Ei(-nu*a/2) )
        F_vd_a = np.exp(dim/2 *np.log(nu*a/2) + nu*a/2) * uG
        dDKL  = ( -nu + (nu+dim)*F_vd_a )/(2*a) # first derivative
        ddDKL = ( (nu+dim)*( (nu/2 +dim/(2*a))*F_vd_a -nu/2 ) )/(2*a) - dDKL/a #second derivative
        a -= dDKL/ddDKL 
        a = a if a > 1e-10 else 1e-10
    old_KLDmin[(dim,nu)] = a
    return a

def nd_to_str(arr : np.ndarray, shift=0, precision=4, suppress=True) -> str:
    """
    Prints the nd-array `arr` shifted by `shift` positions to the right
    """
    oldoptions = np.get_printoptions()
    np.set_printoptions(linewidth=np.inf, precision=precision, suppress=suppress, threshold=sys.maxsize)
    retstr =  ' '*shift + ('\n'+' '*(shift+arr.ndim-1)).join(str(arr).split('\n '))
    np.set_printoptions(**oldoptions)
    return retstr


def covar_from_ellipse(theta, height, width=1, p=0.393469):
    r"""
    Creates a covariance matrix based on the representation of a ellipse. More accurately,

    .. math::
        \left( \begin{array}{cc} \cos(\theta) & \sin(\theta) \\ -\sin(\theta) & cos(\theta)\end{array} \right) \cdot 
        \left( \begin{array}{cc} h^2 & 0 \\ 0 & w^2 \end{array} \right) \cdot 
        \left( \begin{array}{cc} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & cos(\theta)\end{array} \right) \cdot 

    :param theta: The rotation angle
    :param height: The dimension of the ellipse along the axis given by the rotation angle
    :param width: The dimension of the ellipse along the axis given by the rotation angle shifted 
            by 90°, defaults to 1
    :param p: the confidence probability, i.e. integrating the gaussian density over the interior of
            the ellipse should result in exactly p. Defaults to 0.393469 - the one sigma approach, i.e.
            when the Mahalanobis distance is 1
    """
    R = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    F = stats.chi2.ppf(p, df=2)
    return F*(R.T @ np.diag([height, width]) @ R)

def confidence_ellipse(covar, p=0.393469, out_params=False):
    r"""
    Computes the confidence ellipse out of the covariance matrix of a normal distribution.

    Note that if :math:`X \sim \mathcal{N}(0, \Sigma)` then the Mahalanobis distance 
    :math:`x^\top \Sigma^{-1}x` is :math:`\chi^2_d` distributed, with :math:`\Sigma` being a 
    :math:`d \times d` matrix. Therefore, points :math:`x` on the confidence ellipse have to satisfy 
    :math:`x^\top \Sigma^{-1}x = F^{-1}_{\chi^2_2}(p)`, where :math:`F_{\chi^2_2}(\cdot)` is the 
    cummulative distribution function of the :math:`\chi^2` distribution. 

    With a eigenvector decomposition of :math:`\Sigma` we get eigenvalues :math:`\lambda_1, \lambda_2`
    and a rotation matrix :math:`R(\theta)` (from the normalised eigenvectors) such that 
    :math:`\Sigma^{-1} = R(\theta)\, diag\left(\frac{1}{\lambda_1}, \frac{1}{\lambda_2}\right) 
    R(\theta)^\top`.
    Thus from the condition that :math:`x^\top \Sigma^{-1}x = F^{-1}_{\chi^2_2}(p)` we can conclude
    that any point on the ellipse is given by :math:`R(\theta) \left(x \sqrt{\lambda_1 \cdot 
    F^{-1}_{\chi^2_2}(p)},\ y \sqrt{\lambda_2 \cdot F^{-1}_{\chi^2_2}(p)}\right)^\top`, 
    with :math:`x^2 + y^2 = 1`.

    According to `wolframalpha <https://www.wolframalpha.com/input?i=Eigendecomposition+of+%5B%5Ba%2Cb%5D%2C%5Bb%2Cc%5D%5D>`_,
    the eigenvector decomposition of :math:`\Sigma = \bigl( \begin{smallmatrix}a & b\\ b & c\end{smallmatrix}\bigr)`
    results in :math:`\lambda_1 = \frac{a+c}{2} + \sqrt{\left(\frac{a-c}{2}\right)^2+b^2}` and
    :math:`\lambda_2 = \frac{a+c}{2} - \sqrt{\left(\frac{a-c}{2}\right)^2+b^2}` (where 
    :math:`\lambda_1 \geq \lambda_2`) with eigenvectors :math:`v_1 = \left(\frac{\lambda_1- c}{b}, 1\right)^\top` 
    and :math:`v_2 = \left(\frac{\lambda_2-c}{b}, 1\right)^\top`.
    Since :math:`v_1 = (\kappa\, cos(\theta), \kappa\, sin(\theta))` we can conclude that 
    :math:`\tan(\theta) = \frac{1}{\frac{\lambda_1- c}{b}}` thus :math:`\theta = \tan^{-1}(\frac{b}{\lambda_1-c})`


    :param covar: A 2x2 symmetric matrix
    :param p: the confidence probability, i.e. integrating the gaussian density over the interior of
            the ellipse should result in exactly p. Defaults to 0.393469 - the one sigma approach, i.e.
            when the Mahalanobis distance is 1
    :returns: When ``out_params == True`` the method returns the three parameters 
            :math:`\left(\theta, \sqrt{\lambda_1 \cdot F^{-1}_{\chi^2_2}(p)}, \sqrt{\lambda_2 \cdot 
            F^{-1}_{\chi^2_2}(p)}\right)` otherwise it returns a function :math:`E : [0,1] \to \mathbb{R}^2`
            which gives a parameterisation of the elliptic curve. In the first case, these parameters
            can be used in :meth:`covar_from_ellipse` to recreate the covariance matrix.
    """
    F = stats.chi2.ppf(p, df=2)
    [a,b], [_,c] = covar[:2,:2]
    sq = np.sqrt( (a-c)**2/4 +b**2 )
    lambda1 = (a+c)/2 + sq
    lambda2 = (a+c)/2 - sq
    theta = np.arctan2(b, lambda1-c)
    sq_l1, sq_l2 = np.sqrt(F * lambda1), np.sqrt(F * lambda2)
    if out_params:
        return theta, sq_l1, sq_l2
    costheta, sintheta = np.cos(theta), np.sin(theta)
    def parameterisation(t):
        t = np.asarray(t)
        cost, sint = np.cos(2*np.pi*t), np.sin(2*np.pi*t)
        return np.array([sq_l1*costheta*cost-sq_l2*sintheta*sint, sq_l1*sintheta*cost+sq_l2*costheta*sint])
    return parameterisation

def xpbTDxpb_pdf(maxz, d, b, nu=1, N=1001, M=20):
    r"""
    computes the probability density function of 
    
    .. math:: (x+b)^\top\! D (x+b)

    evaluated at N equidistant points in [0,maxz]
    where D = diag(d) is a diagonal matrix, b a shift vector 
    and x has Student's t-distributiuon :math:`t_{\nu}(0,I)` with degrees of freedom nu.

    M controls the sample density of the heavy-tail of the student-T distribution and
    since these values are computed via a inverse fast fourier transform, accuracy of the
    results scale with N.
    """
    # we need three dimensions arr[:,:,:]
    #  the first axes corresponds to the entries of b and d, but these will be collapsed very soon
    #  the second axes corresponds to the original N sample points j=0,...,N-1 corresponding to the entries of a
    #  the third axes corresponds to the M sampled values of u
    v_u = nu/stats.chi2.ppf(np.arange(1/2/M,1,1/M), df=nu)[None,None,:]
    d = np.asarray(d).reshape((-1,1,1))
    b = np.asarray(b).reshape((-1,1,1))**2
    n = N // 2
    N = 2*n +1
    if maxz is None:
        maxz = 10*( np.sum(d*(b+1)) + 4*np.sqrt(np.sum(d**2 *(2*b+1))) )
    dx = maxz / n

    idx = np.arange(-n, n + 1)[None,:,None]
    t = -2* np.pi * idx / (N * dx)
    # characteristic function evaluated at t
    phi = np.exp(1j *t[0,:,:] *np.sum((d*b) /(1 -2j *t *v_u*d), axis=0))/np.prod(np.sqrt((1 -2j *t *v_u*d)), axis=0)  
    
    p = np.sum(fftshift(ifft(ifftshift(phi,axes=0),axis=0),axes=0).real[n:,:], axis=1)/(M*dx)
    #p[n+1:] += p[n-1::-1]
    #p = p[n:]
    p = np.maximum(p, 0)
    xgrid = idx[0,n:,0] * dx
    return xgrid, p

def xpbTDxpb_cdf(maxz, d, b, nu=1, N=1001, M=20):
    r"""
    computes the cumulative distribution function of 
    
    .. math:: (x+b)^\top\! D (x+b)

    evaluated at N equidistant points in [0,maxz]
    where D = diag(d) is a diagonal matrix, b a shift vector 
    and x has Student's t-distributiuon :math:`t_{\nu}(0,I)` with degrees of freedom nu.
    
    M controls the sample density of the heavy-tail of the student-T distribution and
    since these values are computed via a inverse fast fourier transform, accuracy of the
    results scale with N.
    """
    xgrid, p = xpbTDxpb_pdf(maxz, d, b, nu=nu, N=N, M=M)
    yigrid = np.zeros(xgrid.shape)
    yigrid[1:] += (xgrid[1:]-xgrid[:-1])*(p[1:]+p[:-1])/4
    yigrid[:-1] += yigrid[1:]
    return xgrid, np.cumsum(yigrid)

def inv_nu_xpbTDxpb_pdf(d, b, nu=1, s=0, N=101, M=20, minx=None):
    r"""
    computes the probability density function of 
    
    .. math:: \frac{1}{s + (x+b)^\top\! D (x+b)}

    evaluated at N non-equidistant points in [minx,:math:`\tfrac{1}{nu}`]
    where D = diag(d) is a diagonal matrix, b a shift vector 
    and x has Student's t-distributiuon :math:`t_{\nu}(0,I)` with degrees of freedom nu.

    M controls the sample density of the heavy-tail of the student-T distribution and
    since these values are computed via a inverse fast fourier transform, accuracy of the
    results scale with N.
    """
    if minx is None:
        minx = 0.01/s
    maxz = 1/minx - s
    xgrid, p = xpbTDxpb_pdf(maxz, d, b, nu=nu, N=N, M=M)
    ygrid = 1/(s+xgrid[::-1])
    return ygrid, p[::-1]/ygrid**2

def inv_nu_xpbTDxpb_cdf(d, b, nu=1, s=0, N=101, M=20, minx=None):
    r"""
    computes the cumulative distribution function of 
    
    .. math:: \frac{1}{s + (x+b)^\top\! D (x+b)}

    evaluated at N non-equidistant points in 0 and [minx,:math:`\tfrac{1}{nu}`]
    where D = diag(d) is a diagonal matrix, b a shift vector 
    and x has Student's t-distributiuon :math:`t_{\nu}(0,I)` with degrees of freedom nu.
    
    M controls the sample density of the heavy-tail of the student-T distribution and
    since these values are computed via a inverse fast fourier transform, accuracy of the
    results scale with N.
    """
    xgrid, ygrid = inv_nu_xpbTDxpb_pdf(d=d,b=b,nu=nu,s=s,N=N,M=M,minx=minx)
    yigrid = np.zeros(ygrid.shape)
    yigrid[1:] += (xgrid[1:]-xgrid[:-1])*(ygrid[1:]+ygrid[:-1])/4
    yigrid[:-1] += yigrid[1:]
    yigrid = np.cumsum(yigrid)
    return np.concatenate(([0],xgrid)),np.concatenate(([0],yigrid+(1-yigrid[-1])))

def inv_nu_xpbTDxpb_ppf(p, d, b, nu=1, s=0, N=101, M=20, minx=None):
    r"""
    computes the probability point function of 
    
    .. math:: \frac{1}{s + (x+b)^\top\! D (x+b)}

    that is it computes the values :math:`x_i` for which

    .. math:: \mathbb{ \frac{1}{\nu + (x+b)^\top\! D (x+b)} \leq x_i } = p_i

    where D = diag(d) is a diagonal matrix, b a shift vector 
    and x has Student's t-distributiuon :math:`t_{\nu}(0,I)` with degrees of freedom nu.

    M controls the sample density of the heavy-tail of the student-T distribution and
    since these values are computed via a inverse fast fourier transform, accuracy of the
    results scale with N and the closer minx is to zero.
    """
    xgrid, ygrid = inv_nu_xpbTDxpb_cdf(d=d,b=b,nu=nu,s=s,N=N,M=M,minx=minx)
    ps = np.atleast_1d(p).flatten()
    if np.any(ps > 1) or np.any(ps < 0):
        raise ValueError("only percentiles between 0 and 1 allowed")
    idxs = np.searchsorted(ygrid, ps, side='left')
    mask = (idxs > 0)
    xs = xgrid[idxs]
    ws = (ps[mask]-ygrid[idxs[mask]-1])/(ygrid[idxs[mask]]-ygrid[idxs[mask]-1])
    xs[mask] = ws*xs[mask] + (1-ws)*xgrid[idxs[mask]-1]
    return xs

def inv_nu_xpmTBxpm_ppf(p, m, A, B, nu=1, s=0, N=101, M=20, minx=None):
    r"""
    computes the probability point function of 
    
    .. math:: \frac{1}{\nu + (x+m)^\top\! B (x+m)}

    that is it computes the values :math:`x_i` for which

    .. math:: \mathbb{ \frac{1}{\nu + (x+b)^\top\! D (x+b)} \leq x_i } = p_i

    where x has Student's t-distributiuon :math:`t_{\nu}(m,A)` with degrees of freedom nu, 
    and B is a matrix.

    M controls the sample density of the heavy-tail of the student-T distribution and
    since these values are computed via a inverse fast fourier transform, accuracy of the
    results scale with N and the closer minx is to zero.
    """
    sqrtA = np.linalg.cholesky(A)
    d,U = np.linalg.eigh(sqrtA.T @ B @ sqrtA)
    b = U.T @ solve_triangular(sqrtA, m, lower=True)
    return inv_nu_xpbTDxpb_ppf(p=p, d=d, b=b, nu=nu, s=s, N=N, M=M, minx=minx)
