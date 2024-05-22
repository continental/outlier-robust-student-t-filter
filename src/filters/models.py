# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Defines the  motion models
"""

from __future__ import annotations
from typing import Callable

from . import abstract
from . import distributions
from .. import utils
import numpy as np
import sympy as sym

from . import basic

print("-   Loading File 'models.py'")



class LinearDynamicModel(abstract.AbstractDynamicModel):
    r"""
    Dynamic model in which all transformations are linear,

    .. math::
        x_t &= F_t x_{t-1} + v_t, &\quad v_t \sim \mathcal{N}(0, Q_t)\\
        y_t &= H_t x_t + e_t, &\quad e_t \sim \mathcal{N}(0, R_t)

    Note that the linear models can depend on the timestep. Also not that if non-Gaussian distributions
    shall be used, they are still expected to be initialised by their covariance matrix
    """
    def __init__(self, F: np.ndarray | Callable[[float | np.ndarray], np.ndarray],
                 Q: np.ndarray | Callable[[float | np.ndarray], np.ndarray],
                 H: np.ndarray | Callable[[float | np.ndarray], np.ndarray],
                 R: np.ndarray | Callable[[float | np.ndarray], np.ndarray],
                 seed: int = None, state_dim: int = None, obs_dim: int = None,
                 F_dt_dep: bool = True, Q_dt_dep: bool = True,
                 Q_distr: str | Callable[[np.ndarray], abstract.AbstractDistribution] = "normal",
                 R_distr: str | Callable[[np.ndarray], abstract.AbstractDistribution] = "normal") -> None:
        r"""
        Initialises the linear model

        :param state_dim: The dimension of the state vectors :math:`n`
        :type state_dim: int
        :param obs_dim: The dimension of the observation vectors :math:`m`
        :type obs_dim: int
        :param F: State transition matrix :math:`F_t`, a :math:`n \times n` matrix, or if time 
                dependent, a callable function with a single argument returning :math:`n \times n`
                matrices. If :math:`F_{\Delta t}` is instead dependent on the time difference since 
                the last time-step, :obj:`F_dt_dep` has to be set to ``True``.
        :type F: (:math:`n`, :math:`n`) numpy.ndarray or function: float :math:`\to` (:math:`n`, :math:`n`) 
                numpy.ndarray
        :param Q: Process noise covariance matrix :math:`Q_t`, a :math:`n \times n` matrix, or if 
                time dependent, a callable function with a single argument returning 
                :math:`n \times n` matrices. If :math:`Q_{\Delta t}` is instead dependent on the 
                time difference since the last time-step, :obj:`Q_dt_dep` has to be set to ``True``.
        :type Q: (:math:`n`, :math:`n`) numpy.ndarray or function: float :math:`\to` (:math:`n`, :math:`n`) 
                numpy.ndarray
        :param H: Observation model :math:`H_t`, a :math:`m \times n` matrix, or if time dependent, 
                a callable function with a single argument returning :math:`m \times n` matrices.
        :type H: (:math:`m`, :math:`n`) numpy.ndarray or function: float :math:`\to` (:math:`m`, :math:`n`) 
                numpy.ndarray
        :param R: Observation noise covariance matrix :math:`R_t`, a :math:`m \times m` matrix, or 
                if time dependent, a callable function with a single argument returning 
                :math:`m \times m` matrices.
        :type R: (:math:`m`, :math:`m`) numpy.ndarray or function: float :math:`\to` (:math:`m`, :math:`m`) 
                numpy.ndarray
        :param F_dt_dep: Boolean indicating if ``True`` that :math:`F_t` is not dependent on the 
                current time :math:`t`, but on the time-difference since the last timestep 
                :math:`\Delta t`.
        :type F_dt_dep: bool
        :param Q_dt_dep: Boolean indicating if ``True`` that :math:`Q_t` is not dependent on the 
                current time :math:`t`, but on the time-difference since the last timestep 
                :math:`\Delta t`.
        :type Q_dt_dep: bool
        :param Q_distr: Which distribution the process noise has. Either "normal" for a normal 
                distribution, "studentT" for a Student-T distribution with degrees of freedom 
                :math:`\nu=1`, or a tuple ("studentT", :math:`\nu`). 
                Otherwise, it has to be a function taking in :math:`Q` and returning 
                the process noise distribution extending :class:`AbstractDistribution`.
        :type Q_distr: str
        :param R_distr: Which distribution the observation noise has. Either "normal" for normal 
                distribution, "studentT" for a Student-T distribution with degrees of freedom 
                :math:`\nu=1`, or a tuple ("studentT", :math:`\nu`). 
                Otherwise, it has to be a function taking in :math:`R` and returning 
                the process noise distribution extending :class:`AbstractDistribution`.
        :type R_distr: str
        """
        super().__init__(F, Q, H, R, seed, state_dim, obs_dim)
        # The _timed booleans indicate whether the actual time is the argument of the functions
        # the _dt_dep booleans indicate whether the time difference is the argument
        self._F_timed = callable(F) and not F_dt_dep
        self._F_dt_dep = F_dt_dep
        self._Q_timed = callable(Q) and not Q_dt_dep
        self._Q_dt_dep = Q_dt_dep
        self._H_timed = callable(H)
        self._R_timed = callable(R)


        # Test whether all matrices have consistent dimensions
        H_test = H(0) if callable(H) else H
        self.state_dim = H_test.shape[1] if self.state_dim is None else self.state_dim
        self.obs_dim = H_test.shape[0] if self.obs_dim is None else self.obs_dim
        assert H_test.shape == (self.obs_dim,   self.state_dim), (
               f"H should produce ({self.obs_dim  }, {self.state_dim}) matrices, but does {H_test.shape}")
        F_test = F(0) if callable(F) else F
        assert F_test.shape == (self.state_dim, self.state_dim), (
               f"F should produce ({self.state_dim}, {self.state_dim}) matrices, but does {F_test.shape}")
        Q_test = Q(0) if callable(Q) else Q
        assert Q_test.shape == (self.state_dim, self.state_dim), (
               f"Q should produce ({self.state_dim}, {self.state_dim}) matrices, but does {Q_test.shape}")
        R_test = R(0) if callable(R) else R
        assert R_test.shape == (self.obs_dim,   self.obs_dim), (
               f"R should produce ({self.obs_dim  }, {self.obs_dim  }) matrices, but does {R_test.shape}")
        
        if Q_distr == "normal":
            def sQ_distr(iQ):
                return distributions.NormalDistribution(
                    seed = self._rng_gen, mu = np.zeros((self.state_dim)), P = iQ,
                    )
            Q_distr = sQ_distr
        elif Q_distr == "studentT" or (hasattr(Q_distr, "__getitem__") and Q_distr[0] == "studentT"):
            nu = 1 if Q_distr == "studentT" else Q_distr[1]
            def sQ_distr(iQ):
                return distributions.StudentTDistribution(
                    seed = self._rng_gen, mu = np.zeros((self.state_dim)), P = utils.KLDmin_Norm_T(self.state_dim,nu)*iQ,
                    nu = nu, P_covar=True,
                    )
            Q_distr = sQ_distr
        self._Q_distr = Q_distr

        if R_distr == "normal":
            def sR_distr(iR):
                return distributions.NormalDistribution(
                        seed = self._rng_gen, mu = np.zeros((self.obs_dim)), P = iR,
                        )
            R_distr = sR_distr
        elif R_distr == "studentT" or (hasattr(R_distr, "__getitem__") and R_distr[0] == "studentT"):
            nu = 1 if R_distr == "studentT" else R_distr[1]
            def sR_distr(iR):
                return distributions.StudentTDistribution(
                    seed = self._rng_gen, mu = np.zeros((self.obs_dim)), P = utils.KLDmin_Norm_T(self.obs_dim,nu)*iR,
                    nu = nu, P_covar=True, # Note, the different results in a different submission stem from this wrongly being set to False
                    )
            R_distr = sR_distr
        self._R_distr = R_distr

    def forecast(self, mu: np.ndarray = None, P: np.ndarray = None, next_time: float = None,
                  dt: float = None) -> tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, np.ndarray):
        r"""
        Predicts model behaviour at given state estimates.

        Given state estimates as Gaussian distributions, give estimates to the states at the 
        next timestep, their covariance matrices, the 
        expected observations and their covariance matrices, as well as the covariance matrices of 
        the additive process- and observation-noises. 
        In the following :math:`n = {}`:attr:`state_dim`, :math:`m = {}`:attr:`obs_dim` and :math:`k` is
        the number of points the model should be evaluated at

        :param mu: Means of current state estimates, of shape :math:`(n,), (n,1), (k,n)` or or :math:`(k,n,1)`.
                In the later case, we also need a stack of :math:`k` covariance matrices.
        :type mu: numpy.ndarray
        :param P: Covariance matrices of the current state, either a matrix of shape 
                :math:`(n,n)` or a stack of matrices of shape :math:`(k,n,n)`.
        :type P: numpy.ndarray
        :param time: The current time, depending on whether the model requires this parameter.
        :type time: float
        :param dt: The time difference between the current :obj:`time` and next time step :obj:`time + dt`,
                depending on whether the model requires this parameter.
        :type dt: float
        :raises NotImplementedError: The function is not implemented in the abstract class 
                AbstractDynamicModel.
        :returns:

                .. math::
                    (F \mu, F P F^\top, Q, HF \mu, H F P F^\top H^\top, H Q H^\top, R, 
                     F P F^\top H^\top, Q H^\top)

                of next estimated transitioned state means :math:`F \mu`, estimated transitioned
                state covariances :math:`F P F^\top`, process noise covariance matrix :math:`Q`,
                expected observations :math:`HF \mu`, expected observation noise covariance
                :math:`H F P F^\top H^\top` induced by the state uncertainty, expected observation
                noise covariance :math:`H Q H^\top` induced by the process noise, pure observation
                noise covariance matrix :math:`R`, cross correlation between state estimate and
                observation :math:`F P F^\top H^\top`, as well as cross correlation between process
                noise and observation :math:`Q H^\top`.
        :rytpe: list of numpy.ndarray of shapes: :math:`(\,(k,n,1), (k,n,n), (k,n,n), (k,m,1), 
                (k,m,m), (k,m,m), (k,m,m), (k,n,m), (k,n,m) \,)`
        """
        # Check wether arguments 'dt' and 'time' are needed and provided
        assert (dt is not None) or ((not self._F_dt_dep) and (not self._Q_dt_dep)), (
                "Argument 'dt' is required since " + 
                ", ".join(np.array(["F", "Q"])[[self._F_dt_dep, self._Q_dt_dep]]) +
                " depend(s) on it.")
        assert (next_time is not None) or not (self._F_timed or self._Q_timed or
                                          self._H_timed or self._R_timed), (
                "Argument 'time' is required since " +
                ", ".join(np.array(["F", "Q", "H", "R"])[[self._F_timed, self._Q_timed, self._H_timed, self._R_timed]])
                + " require(s) it")

        # Check that mu and P have the correct dimensions
        n = self.state_dim
        mu = np.atleast_1d(np.squeeze(mu))
        mu = mu[:,:,None] if mu.ndim == 2 else mu[None,:,None]
        k, n_mu = mu.shape[:2]
        assert n == n_mu, (f"'mu' should have shape ({n},), ({n},1), (k,{n}) or (k,{n},1)" +
                                                   f" for some k, but has shape {mu.shape}")
        assert P.ndim in [2,3], f"P's number of dimensions should be 2 or 3 (matrix or tensor), but has {P.ndim}."
        if k == 1:
            assert (P.shape[-2:] == (n,n)), (
                f"'P' should have shape (1, {n}, {n}) or ({n}, {n}) but has {P.shape}")
            P = P[None, :, :] if P.ndim == 2 else P
        else:
            assert (P.shape == (k,n,n)), (
                f"'P' should have shape ({k}, {n}, {n}) but has {P.shape}")

        F = self._F(next_time) if self._F_timed else (
            self._F(dt) if self._F_dt_dep else self._F)
        F = F[None, :, :] if k == 1 else np.repeat(F[None,:,:], k, axis=0)
        Q = self._Q(next_time) if self._Q_timed else (
            self._Q(dt) if self._Q_dt_dep else self._Q)
        Q = Q[None, :, :] if k == 1 else np.repeat(Q[None,:,:], k, axis=0)
        H = self._H(next_time) if self._H_timed else self._H
        H = H[None, :, :] if k == 1 else np.repeat(H[None,:,:], k, axis=0)
        R = self._R(next_time) if self._R_timed else self._R
        R = R[None, :, :] if k == 1 else np.repeat(R[None,:,:], k, axis=0)
        Fm = F @ mu
        FT = np.transpose(F, (0,2,1))
        HT = np.transpose(H, (0,2,1))
        PF = P @ FT
        FPF = F @ PF
        PFH = PF @ HT
        FPFH = FPF @ HT
        QH = Q @ HT
        #m = self.obs_dim
        #print(f"returns:\n {Fm.shape}, {FPF.shape}, {Q.shape}, {(H @ Fm).shape}, {(H @ FPFH).shape},"+
        #      f" {(H @ QH).shape}, {R.shape}, {FPFH.shape}, {QH.shape}"+
        #      f"\nbut should:\n ({k}, {n}, 1), ({k}, {n}, {n}), ({k}, {n}, {n}), ({k}, {m}, 1), "+
        #      f"({k}, {m}, {m}), ({k}, {m}, {m}), ({k}, {m}, {m}), ({k}, {n}, {m}), ({k}, {n}, {m})")
        return (Fm, FPF, Q, H @ Fm, H @ FPFH, H @ QH, R, FPFH, QH, PF, PFH)
    
    def forecast_smoother(self, mu: np.ndarray = None, P: np.ndarray = None, next_time: float = None,
                dt: float = None) -> tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        
        # Check wether arguments 'dt' and 'time' are needed and provided
        assert (dt is not None) or ((not self._F_dt_dep) and (not self._Q_dt_dep)), (
                "Argument 'dt' is required since " + 
                ", ".join(np.array(["F", "Q"])[[self._F_dt_dep, self._Q_dt_dep]]) +
                " depend(s) on it.")
        assert (next_time is not None) or not (self._F_timed or self._Q_timed or
                                          self._H_timed or self._R_timed), (
                "Argument 'time' is required since " +
                ", ".join(np.array(["F", "Q", "H", "R"])[[self._F_timed, self._Q_timed, self._H_timed, self._R_timed]])
                + " require(s) it")

        # Check that mu and P have the correct dimensions
        n = self.state_dim
        new_mu = mu.reshape((-1,1))
        n_mu = mu.shape[0]
        assert n == n_mu, (f"'mu' should have shape ({n},) or ({n},1)" +
                                                   f" for some k, but has shape {mu.shape}")
        assert P.shape == (n,n), f"P should be a {n}x{n} matrix but has shape {P.shape}."

        F = self._F(next_time) if self._F_timed else (
            self._F(dt) if self._F_dt_dep else self._F)
        Q = self._Q(next_time) if self._Q_timed else (
            self._Q(dt) if self._Q_dt_dep else self._Q)
        Fm = F @ new_mu
        PF = P @ F.T
        FPF = F @ PF
        return (Fm, FPF, Q, PF)

    def sample(self, x: np.ndarray, t0: float, dt: float | np.ndarray, give_noise: bool = False
               ) -> tuple(np.ndarray, np.ndarray) | tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        r"""
        Samples possible next future states and observations. To get distributions, see :meth:`forecast`.

        :param x: Means of current state estimates, either a single mean of shape (:attr:`state_dim`,) 
                or (:attr:`state_dim`, :math:`k`) for :math:`k` consecutive means.
        :type x: numpy.ndarray
        :param t0: The initial time.
        :type t0: float
        :param dt: The time differences between the sample timesteps depending on whether the model
                requires this parameter.
        :type dt: float or a numpy.ndarray
        :raises NotImplementedError: The function is not implemented in the abstract class 
                AbstractDynamicModel.
        :returns: sampled possible next states and observations :math:`(\mu_{t+1}, \tilde y_{t+1})`,
            more precisely :math:`(F(\nu_{t}+\tilde v_{t+1}, H(F(\nu_{t}+\tilde v_{t+1}) +\tilde e_t)`.
            or a tuple :math:`([\mu_{t}, ..., \mu_{t+T}], [\tilde y_{t+1}, ..., \tilde y_{t+T}])`,
            i.e. a (:math:`T`, :attr:`state_dim`, :math:`k`) and a (:math:`T`, :attr:`state_obs`, :math:`k`)
            shaped array, if an array of times/time differences is given, otherwise a 
            (:attr:`state_dim`, :math:`k`) and a (:attr:`state_obs`, :math:`k`)

            Note that it will also return the initial :math:`\mu_t` value such that there is one 
            more :math:`\mu` value than :math:`\tilde y` value.
        """
        dt = np.atleast_1d(dt)
        T = len(dt)
        times = np.empty((T+1,))
        times[0] = t0
        for i in range(T):
            times[i+1] = times[i] + dt[i]

        n, m = (self.state_dim, self.obs_dim)
        assert x.shape[0] == n and x.ndim <= 2, (f"'mu' should have shape ({n},) or ({n},k)" +
                                                 f" for some k, but has shape {x.shape}")
        k = x.shape[1] if x.ndim == 2 else 1
        x = x.reshape((n, k))

        mu_new, y_new = (np.zeros((T+1, n, k)), np.zeros((T, m, k)))
        vs, es        = (np.zeros((T, n, k)), np.zeros((T, m, k)))
        mu_new[0, :, :] = x
        for i in range(T):
            F = self._F(times[i+1]) if self._F_timed else (
                self._F(dt[i]) if self._F_dt_dep else self._F)
            Q = self._F(times[i+1]) if self._Q_timed else (
                self._Q(dt[i]) if self._Q_dt_dep else self._Q)
            H = self._H(times[i+1]) if self._H_timed else self._H
            R = self._R(times[i+1]) if self._R_timed else self._R
            vs[i,:,:] = self._Q_distr(Q).rvs(m=k)[0]
            es[i,:,:] = self._R_distr(R).rvs(m=k)[0]
            #v = scipy.stats.multivariate_normal.rvs(cov=Q, size=k, random_state=self._rng_gen)
            #e = scipy.stats.multivariate_normal.rvs(cov=R, size=k, random_state=self._rng_gen)
            mu_new[i+1, :, :] = F @ mu_new[i, :, :] + vs[i,:,:]
            y_new[i, :, :] = H @ mu_new[i+1, :, :] + es[i,:,:]
        if give_noise:
            return ((mu_new), (y_new), (vs), (es))
        return ((mu_new), (y_new))
    
    def __copy__(self) -> abstract.AbstractDynamicModel:
        copy = super().__copy__()
        copy._F_dt_dep = self._F_dt_dep
        copy._Q_dt_dep = self._Q_dt_dep
        copy._Q_distr  = self._Q_distr
        copy._R_distr  = self._R_distr
        return copy


def extendedSinger_transition(alpha, beta, d=2):
    r"""
    Returns a function that takes a time-difference :math:`\Delta t` and returns a matrix 
    :math:`F_{\Delta t}` such that for arbitrary :math:`x_0` we have that 
    :math:`F_{\Delta t}x_0 = x(\Delta t)`, where :math:`x` is the 
    solution to the ODE:
    
    .. math:: 
        \dot p(t) &= v(t),\\
        \dot v(t) &= - \beta \cdot v(t) +a(t),\\
        \dot a(t) &= - \alpha \cdot a(t),
    
    and :math:`x(0) = x_0`.

    :param alpha: the parameter :math:`\alpha > 0`.
    :type alpha: float
    :param beta: the parameter :math:`\beta \geq 0`.
    :type beta: float
    :param d: the dimension of the problem, i.e. in how many dimensions the ODE above should hold. 
            For ``d = 1`` we have 1-d movement with three state variables :math:`(p_x, v_x, a_x)`, 
            for ``d = 2`` we have 2-d movement with six state variables 
            :math:`(p_x, p_y, v_x, v_y, a_x, a_y)` and so on.
    :type d: int
    :returns: the matrix :math:`F_{\Delta t}`.
    :rtype: function (float,) :math:`\to` :math:`3d \times 3d` numpy array
    """
    assert alpha > 0, f"alpha has to be greater than zero, but is {alpha}."
    assert beta >= 0, f"beta has to be non-negative, but is {beta}."

    if abs(beta) > 1e-10:
        if abs(beta-alpha)/alpha > 1e-6:
            def F(dt):
                ea, eb = (np.exp(-alpha*dt), np.exp(-beta*dt))
                eam1, ebm1 = (np.expm1(-alpha*dt), np.expm1(-beta*dt))
                return np.kron(np.array([
                    [1, -ebm1/beta, np.abs((ebm1*alpha-eam1*beta)/((beta-alpha)*alpha*beta))],
                    [0, eb,            (ea - eb)/(beta-alpha)                   ],
                    [0, 0,             ea                                       ]
                ]), np.eye(d))
        else:
            def F(dt):
                ea, eb = (np.exp(-alpha*dt), np.exp(-beta*dt))
                eam1, ebm1 = (np.expm1(-alpha*dt), np.expm1(-beta*dt))
                return np.kron(np.array([
                    [1, -ebm1/beta, (-eam1-ea*(alpha*dt))/alpha**2],
                    [0, eb,            dt*ea                        ],
                    [0, 0,             ea                           ]
                ]), np.eye(d))
    else:
        def F(dt):
            ea, eam1 = (np.exp(-alpha*dt), np.expm1(-alpha*dt))
            return np.kron(np.array([
                [1, dt, dt/alpha +eam1/alpha**2],
                [0, 1,  -eam1/(alpha)            ],
                [0, 0,  ea                        ]
            ]), np.eye(d))
    return F


def extendedSinger_noise(alpha, beta, sigma2, d=2):
    r"""
    Returns a function that takes a time-difference :math:'\Delta t' and returns a matrix 
    :math:`Q_{\Delta t}` such that the solution :math:`x^*(t)` to the stochastic differential equation:
    
    .. math:: 
        \dot p(t) &= v(t),\\
        \dot v(t) &= - \beta \cdot v(t) +a(t), \quad\quad x^*(0) = x_0\\
        \dot a(t) &= - \alpha \cdot a(t) +\sigma w(t),
    
    with white noise :math:`w(t)`, is a random variable with :math:`x^*(\Delta t) \sim \mathcal{N}
    \big(F_{\Delta t}\ x_0, Q_{\Delta t}\big)`, where :math:`F_{\Delta t}` is computed by 
    :meth:`extendedSinger_transition`.

    :param alpha: the parameter :math:`\alpha > 0`.
    :type alpha: float
    :param beta: the parameter :math:`\beta \geq 0`.
    :type beta: float
    :param sigma2: the variance of the white noise :math:`\sigma^2 > 0`.
    :type sigma2: float
    :param d: the dimension of the problem, i.e. in how many dimensions the ODE above should hold. 
            For ``d = 1`` we have 1-d movement with three state variables :math:`(p_x, v_x, a_x)`, 
            for ``d = 2`` we have 2-d movement with six state variables 
            :math:`(p_x, p_y, v_x, v_y, a_x, a_y)` and so on.
    :type d: int
    :type timespan: float
    :returns: the matrix :math:`Q_{\Delta t}`.
    :rtype: function (float,) :math:`\to` :math:`3d \times 3d` numpy array
    """
    assert alpha > 0, f"alpha has to be greater than zero, but is {alpha}."
    assert beta >= 0, f"beta has to be non-negative, but is {beta}."

    if True:
        a = sym.Symbol("\alpha")
        b = sym.Symbol("\beta")
        t_diff = sym.Symbol(r"\Delta t\ ")
        if abs(beta) > 1e-10:
            if abs(beta-alpha)/alpha > 1e-6:
                Q00 = ( (
                            -(b**2 *(sym.exp(-2*t_diff*a)-1))/2/a
                            +(a-b)*( 2*a*(sym.exp(-t_diff*b)-1) )/b
                            -(a-b)*( 2*b*(sym.exp(-t_diff*a)-1) )/a
                            -(a**2 *(sym.exp(-2*t_diff*b)-1))/2/b
                            +(2*a*b*(sym.exp(-(a+b)*t_diff)-1))/(a+b)
                        )
                            + (a-b)**2 * t_diff
                        ) / (a**2 * b**2 * (a-b)**2)
                Q01 = ( (a-b) + b *sym.exp(-t_diff*a) - a *sym.exp(-t_diff*b) )**2/2/(a**2 *b**2 *(a-b)**2)
                Q02 = ( a/(a+b)*(sym.exp(-t_diff*(a+b))-1) + (b/a-1)*(sym.exp(-t_diff*a)-1) - b/2/a*(sym.exp(-2*t_diff*a)-1) ) / (a*b*(a-b))
                Q11 = ((1-sym.exp(-2*b*t_diff))/2/b - 2*(1-sym.exp(-(a+b)*t_diff))/(a+b) + (1-sym.exp(-2*a*t_diff))/2/a) / (a-b)**2
                Q12 = ( (sym.exp(-(a+b)*t_diff)-1)/(a+b) - (sym.exp(-2*a*t_diff)-1)/(2*a) )/(b-a)
            else:
                Q00 = (4*t_diff*a - 11 + 8*(t_diff*a + 2)*sym.exp(-t_diff*a) - (2*t_diff**2 *a**2 + 6*t_diff*a+5)*sym.exp(-2*t_diff*a))/(4*a**5)
                Q01 = (1- (t_diff*a+1)*sym.exp(-t_diff*a))**2 / (2*a**4)
                Q02 = ( (1-2*sym.exp(-t_diff*a))**2 - (1-2*t_diff*a)*sym.exp(-2*t_diff*a) )/(4*a**3)
                Q11 = ( 1- (2*t_diff**2*a**2 + 2*t_diff*a + 1)*sym.exp(-2*t_diff*a) ) / (4*a**3)
                Q12 = ( 1 - (2*t_diff*a+ 1)* sym.exp(-2*t_diff*a) ) / (4*a**2)
        else:
                Q00 = ( 5 - 2*(1 - t_diff*a)**3 - 12*t_diff*a*sym.exp(-t_diff*a) - 3*sym.exp(-2*t_diff*a) )/(6*a**5)
                Q01 = ( (1 - t_diff*a - sym.exp(-t_diff*a))**2 )/(2*a**4)
                Q02 = ( (1 - 2*t_diff*a*sym.exp(-t_diff*a) - sym.exp(-2*t_diff*a)) )/(2*a**3)
                Q11 = ( (2*t_diff*a - 3 + 4*sym.exp(-t_diff*a) - sym.exp(-2*t_diff*a)) )/(2*a**3)
                Q12 = ( (1 - sym.exp(-t_diff*a))**2 )/(2*a**2)
        Q22 = 1/2/a - sym.exp(-2*t_diff*a)/2/a
        symQ = sym.Matrix([[Q00, Q01, Q02],
                            [0, Q11, Q12],
                            [0, 0, Q22]])
        #symQ = symQ.subs(a,alpha).subs(b,beta)
        #symQ = sym.lambdify(t_diff, symQ, "numpy")
    
        oldQs = {}
        def Q(dt):
            if dt in oldQs.keys():
                return oldQs[dt]
            # precision 4
            Qs = np.array(symQ.evalf(4, subs={a: alpha, b: beta, t_diff: dt}), dtype=float)
            #Qs = symQ(dt)
            Qs[1,0] = Qs[0,1]
            Qs[2,0] = Qs[0,2]
            Qs[2,1] = Qs[1,2]
            Qs = np.kron(Qs, sigma2*np.eye(d))
            oldQs[dt] = Qs
            return Qs
        return Q

    else:
        a = alpha
        b = beta
        def Q(dt):
            # The following description was found by sorting the anti-derivative of FHHF by their exponents
            # Then in the integral terms of the form (exp(-a*dt)-1) appear, which can be expressed by 
            # power series (for Q[0,0]). Collecting the coefficients of all power series leads to the following
            # truncated series
            # And since these terms do not divide by zero, they can be taken independent of whether
            # a=0, b=0 or a=b
            at = a* dt
            
            Qs= np.array([
                [
                    (   +1/20     # (-1)^(k+1) /k! ( b^2 *(2a)^(k-1) +a^2 *(2b)^(k-1) +2*(a-b)*( b*a^(k-1) - a*b^(k-1) ) -2ab(a+b)^(k-1) )/(a^2*b^2 (a-b)^2)
                        -1/36    *(a+b)*dt 
                        +1/504   *(5*a**2 +7*a*b +5*b**2)*dt**2
                        -1/2880  *(8*a**3 +13*a**2*b +13*a*b**2 +8*b**3)*dt**3
                        +1/25920 *(17*a**4 +30*a**3*b +35*a**2*b**2 +30*a*b**3 +17*b**4)*dt**4
                        -1/302400*(41*a**5 +76*a**4*b +97 *a**3*b**2 +97*a**2*b**3 +76*a*b**4 +41*b**5)*dt**5
                        +1/6652800*(167*a**6 +319*a**5*b +431*a**4*b**2 +473*a**3*b**3 +431*a**2*b**4 +319*a*b**5 +167*b**6)*dt**6
                        -1/21772800*(a+b)*(92*a**6 +87*a**5*b +164*a**4*b**2 +129*a**3*b**3 +164*a**2*b**4 +87*a*b**5 +92*b**6)*dt**7
                        +1/283046400*(185*a**8 +364*a**7*b +523*a**6*b**2 +637*a**5*b**3 +679*a**4*b**4 +637*a**3*b**5 +523*a**2*b**6 +364*a*b**7 +185*b**8)*dt**8
                        -1/3353011200*(a+b)*(314*a**8 +308*a**7*b +600*a**6*b**2 +539*a**5*b**3 +732*a**4*b**4 +539*a**3*b**5 +600*a**2*b**6 +308*a*b**7 +314*b**8)*dt**9
                    ) *dt**5,
                    #(b*np.expm1(-a*dt) -a*np.expm1(-b*dt))**2/2/(a**2 *b**2 *(a-b)**2),
                    (   +1/4      # (-1)^k /k! (a^(k-1) -b^(k-1))/2/(a-b)
                        -1/12*    (a+b)*dt
                        +1/48*    (a**2 +a*b +b**2)*dt**2
                        -1/240*   (a+b)*(a**2+b**2)*dt**3
                        +1/1440*  (a**4 +a**3*b +a**2*b**2 +a*b**3+b**4)*dt**4
                        -1/10080* (a+b)*(a**2 -a*b +b**2)*(a**2 +a*b +b**2)*dt**5
                        +1/80640* (a**6 +a**5*b +a**4*b**2 +a**3*b**3 +a**2*b**4 +a*b**5 +b**6)*dt**6
                        -1/725760*(a+b)*(a**2 +b**2)*(a**4 +b**4)*dt**7
                        +1/7257600*(a**2 +a*b +b**2)*(a**6 +a**3*b**3 +b**6)*dt**8
                        -1/79833600*(a+b)*(a**4 -a**3*b +a**2*b**2 -a*b**3 +b**4)*(a**4 +a**3*b +a**2*b**2 +a*b**3 +b**4)*dt**9
                    )**2*2 *dt**4,
                    (   +1/6      # (-1)^k /k! (a*(a+b)^(k-1) +(b-a) * a^(k-1) - b*(2a)^(k-1))/(a*b*(a-b))
                        -1/24*    (4*a+b)*dt
                        +1/120*   (11*a**2 +5*a*b +b**2)*dt**2
                        -1/720*   (26*a**3 +16*a**2*b +6*a*b**2 +b**3)*dt**3
                        +1/5040*  (57*a**4 +42*a**3*b +22*a**2*b**2 +7*a*b**3 +b**4)*dt**4
                        -1/40320* (120*a**5 +99*a**4*b +64*a**3*b**2 +29*a**2*b**3 +8*a*b**4 +b**5)*dt**5
                        +1/362880*(247*a**6 +219*a**5*b +163*a**4*b**2 +93*a**3*b**3 +37*a**2*b**4 +9*a*b**5 +b**6)*dt**6
                        -1/3628800*(502*a**7 +466*a**6*b +382*a**5*b**2 +256*a**4*b**3 +130*a**3*b**4 +46*a**2*b**5 +10*a*b**6 +b**7)*dt**7
                        +1/39916800*(1013*a**8 +968*a**7*b +848*a**6*b**2 +638*a**5*b**3 +386*a**4*b**4 +176*a**3*b**5 +56*a**2*b**6 +11*a*b**7 +b**8)*dt**8
                        -1/479001600*(-2036*a**9 +1981*a**8*b +1816*a**7*b**2 +1486*a**6*b**3 +1024*a**5*b**4 +562*a**4*b**5 +232*a**3*b**6 +67*a**2*b**7 +12*a*b**8 +b**9)*dt**9
                    ) *dt**3,
                ],
                [
                    0,
                    (   +1/3      # (-1)^k/k! (2(a+b)^(k-1) - (2a)^(k-1) - (2b)^(k-1))/(a-b)^2
                        -1/4*     (a+b)*dt
                        +1/60*    (7*a**2 +10*a*b +7*b**2)*dt**2
                        -1/72*    (3*a**3 +5*a**2*b +5*a*b**2 +3*b**3)*dt**3
                        +1/2520*  (31*a**4 +56*a**3*b +66*a**2*b**2 +56*a*b**3 +31*b**4)*dt**4
                        -1/2880*  (a+b)*(9*a**4 +8*a**3*b +14*a**2*b**2 +8*a*b**3 +9*b**4)*dt**5
                        +1/181440*(127*a**6 +246*a**5*b +337*a**4*b**2 +372*a**3*b**3 +337*a**2*b**4 +246*a*b**5 +127*b**6)*dt**6
                        -1/604800*(a+b)*(85*a**6 +82*a**5*b +155*a**4*b**2 +124*a**3*b**3 +155*a**2*b**4 +82*a*b**5 +85*b**6)*dt**7
                        +1/19958400*(511*a**8 +1012*a**7*b +1468*a**6*b**2 +1804*a**5*b**3 +1930*a**4*b**4 +1804*a**3*b**5 +1468*a**2*b**6 +1012*a*b**7 +511*b**8)*dt**8
                        -1/21772800*(a+b)*(93*a**8 +92*a**7*b +180*a**6*b**2 +164*a**5*b**3 +222*a**4*b**4 +164*a**3*b**5 +180*a**2*b**6 +92*a*b**7 +93*b**8)*dt**9
                    ) *dt**3,
                    #( np.expm1(-(a+b)*dt)/(a+b) - np.expm1(-2*a*dt)/(2*a) )/(b-a),
                    (   +1/2      # (-1)^k/k!  ((a+b)^(k-1) - (2*a)^(k-1))/(b-a) 
                        -1/6*     (3*a+b)*dt
                        +1/24*    (7*a**2 +4*a*b +b**2)*dt**2
                        -1/120*   (3*a+b)*(5*a**2 +2*a*b +b**2)*dt**3
                        +1/720*   (31*a**4 +26*a**3*b +16*a**2*b**2 +6*a*b**3 +b**4)*dt**4
                        -1/5040*  (3*a+b)*(3*a**2 +b**2)*(7*a**2 +4*a*b +b**2)*dt**5
                        +1/40320* (127*a**6 +120*a**5*b +99*a**4*b**2 +64*a**3*b**3 +29*a**2*b**4 +8*a*b**5 +b**6)*dt**6
                        -1/362880*(3*a +b)*(5*a**2 +2*a*b +b**2)*(17*a**4 +4*a**3*b +6*a**2*b**2 +4*a*b**3 +b**4)*dt**7
                        +1/3628800*(7*a**2 +4*a*b +b**2)*(73*a**6 +30*a**5*b +39*a**4*b**2 +28*a**3*b**3 +15*a**2*b**4 +6*a*b**5 +b**6)*dt**8
                        -1/39916800*(3*a +b)*(11*a**4 - 2*a**3*b +4*a**2*b**2 +2*a*b**3 +b**4)*(31*a**4 +26*a**3*b +16*a**2*b**2 +6*a*b**3 +b**4)*dt**9
                    ) *dt**2
                ],
                [
                    0,
                    0,
                    #-np.expm1(-2*a*dt)/2/a, #(-1)^(k+1) /k! (2*a)^(k-1)
                    (+1 -at +2/3*at**2 -at**3/3 +2/15*at**4 - 2/45*at**5 +4/315*at**6 -1/315*at**7 +2/2835*at**8 -2/14175*at**9
                    )* dt,

                ]
            ])
            Qs[1,0] = Qs[0,1]
            Qs[2,0] = Qs[0,2]
            Qs[2,1] = Qs[1,2]
            return np.kron(Qs, sigma2*np.eye(d))
        return Q

def ExtendedSingerModel(
        alpha, beta, sigma2, d, seed: int = None,
        H: np.ndarray | Callable[[float | np.ndarray], np.ndarray] = np.array([[1,0,0]]),
        R: np.ndarray | Callable[[float | np.ndarray], np.ndarray] = np.array([[1]]),
        Q_distr: str | Callable[[np.ndarray], basic.AbstractDistribution] = "normal",
        R_distr: str | Callable[[np.ndarray], basic.AbstractDistribution] = "normal",
        ) -> basic.LinearDynamicModel:
    r"""
    Returns a linear dynamical model described by the stochastic ode with white noise :math:`w(t)`
    
    .. math:: 
        \dot p(t) &= v(t),\\
        \dot v(t) &= - \beta \cdot v(t) +a(t), \\
        \dot a(t) &= - \alpha \cdot a(t) +\sigma w(t),
    
    Given an initial state :math:`x_0 = [p_0, v_0, a_0]^\top`, the state :math:`\Delta t` time-units 
    into the future is a random variable with distribution 
    :math:`x(t_0 + \Delta t) \sim \mathcal{N}(F_{\Delta t}\ x_0, Q_{\Delta t})`, where 
    :math:`F_{\Delta t}` is calculated by :meth:`extendedSinger_transition` and 
    :math:`Q_{\Delta t}` is calculated by :meth:`extendedSinger_noise`.
    
    :param alpha: the parameter :math:`\alpha > 0`.
    :type alpha: float
    :param beta: the parameter :math:`\beta \geq 0`.
    :type beta: float
    :param sigma2: the variance of the white noise :math:`\sigma^2 > 0`.
    :type sigma2: float
    :param d: the dimension of the problem, i.e. in how many dimensions the ODE above should hold. 
            For ``d = 1`` we have 1-d movement with three state variables :math:`(p_x, v_x, a_x)`, 
            for ``d = 2`` we have 2-d movement with six state variables 
            :math:`(p_x, p_y, v_x, v_y, a_x, a_y)` and so on.
    :type d: int
    :param H: Observation model :math:`H_t`, a :math:`m \times n` matrix, or if time dependent, 
            a callable function with a single argument returning :math:`m \times n` matrices.
            If :math:`n = \frac{d}{3}`, :math:`H` will be applied dimension-wise 
    :type H: (:math:`m`, :math:`n`) numpy.ndarray or function: float :math:`\to` (:math:`m`, :math:`n`) 
            numpy.ndarray.
    :param R: Observation noise covariance matrix :math:`R_t`, a :math:`m \times m` matrix, or 
            if time dependent, a callable function with a single argument returning 
            :math:`m \times m` matrices.
            Similarly to :math:`H`, it will be applied dimension-wise if applicable.
    :type R: (:math:`m`, :math:`m`) numpy.ndarray or function: float :math:`\to` (:math:`m`, :math:`m`) 
            numpy.ndarray
    :param Q_distr: Which distribution the process noise has. Either "normal" for a normal 
            distribution, or a function taking in :math:`Q` and returning the process noise 
            distribution extending :class:`AbstractDistribution`.
    :type Q_distr: str
    :param R_distr: Which distribution the observation noise has. Either "normal" for normal 
            distribution or a function taking in :math:`R` and returning the process noise 
            distribution extending :class:`AbstractDistribution`.
    :type R_distr: str
    """
    
    Hm, Hn = H(0).shape if callable(H) else H.shape
    assert (Hn == 3) or (Hn == d*3), f"H should have shape (x,{d*3}) or (x,3) but is ({Hm}, {Hn})"
    if Hn != d*3:
        Hd = (lambda t: np.kron(H(t), np.eye(d))) if callable(H) else np.kron(H, np.eye(d))
        Hm = Hm*d
    else:
        Hd = H
    
    Rm, Rn = R(0).shape if callable(R) else R.shape
    assert (Rm==Rn) and (Rm==Hm or d*Rm==Hm), f"R should have shape ({Hm},{Hm}) or ({Hm/d:.1f},{Hm/d:.1f}) but is ({Rm}, {Rn})"
    if Rm != Hm:
        Rd = (lambda t: np.kron(R(t), np.eye(d))) if callable(R) else np.kron(R, np.eye(d))
        Rm = Rm*d
    else:
        Rd = R
    return LinearDynamicModel(
                F=extendedSinger_transition(alpha, beta, d), H=Hd, R=Rd, seed=seed, 
                Q=extendedSinger_noise(alpha, beta, sigma2, d), 
                state_dim=3*d, obs_dim=None, F_dt_dep=True, Q_dt_dep=True, 
                Q_distr=Q_distr, R_distr=R_distr)