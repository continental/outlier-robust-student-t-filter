# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Defines the basic Filter models like the :class:`KalmanFilter` as references for further 
implementations.
"""

from __future__ import annotations
from typing import Dict
import numpy
import scipy
from .abstract import AbstractDynamicModel, AbstractFilter, AbstractSmoother
from .distributions import NormalDistribution, StudentTDistribution
from .. import utils


class NormalStateFilter(AbstractFilter):
    """
    Abstract Class for all filter that represent their state estimates by normal distributions of 
    the internal attributes ``_state_mean`` and ``_state_covar``.
    """

    _state_distr_class = NormalDistribution

    def _state_distr(self) -> Dict[str, numpy.ndarray]:
        return NormalDistribution.create_params(mu = self._state_mean, P = self._state_covar)


class StudentTStateFilter(AbstractFilter):
    """
    Abstract Class for all filter that represent their state estimates by student-t distributions of 
    the internal attributes ``_state_mean``, ``_state_covar`` (Note that this is not the scale matrix)
    and ``_state_nu``.
    """

    _state_distr_class = StudentTDistribution
    _state_nu : float      = 1              #: The degrees of freedom of the state distribution
    _state_scalar : float  = 1              #: Additional scaling factor on the state covariance matrix, i.e. to compute the scale matrix

    def __init__(self, model: AbstractDynamicModel, mean: numpy.ndarray, covar: numpy.ndarray, state_nu : float = 1,
                 current_time: float = None, seed: int = None, exp_hist_len: int = 100, 
                 process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        """
        Initialises the Filter

        :param state_nu: The degrees of freedom of the Student-t filter estimating the initial state, defaults to 1.
        """
        self._state_nu = state_nu
        super().__init__(model=model, mean=mean, covar=covar, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._state_scalar *= utils.KLDmin_Norm_T(dim=model.state_dim, nu=state_nu)

    def _state_distr(self) -> Dict[str, numpy.ndarray]:
        return StudentTDistribution.create_params(mu = self._state_mean, P = self._state_scalar*self._state_covar,
                                                  nu = self._state_nu)

class NormalStateSmoother(AbstractSmoother):
    """
    Abstract Class for all smoother that represent their state estimates by normal distributions of 
    the internal attributes ``_state_mean`` and ``_state_covar``.
    """
    
    _state_distr_class = NormalDistribution

    def _state_distr(self) -> Dict[str, numpy.ndarray]:
        return NormalDistribution.create_params(mu = self._state_mean, P = self._state_covar)


class StudentTStateSmoother(AbstractSmoother):
    """
    Abstract Class for all smoother that represent their state estimates by student-t distributions of 
    the internal attributes ``_state_mean``, ``_state_covar`` (Note that this is the covariance matrix
    that minimises the Kulback-Leibler divergence to the internally used scale matrix, they differ
    by the scalar factor :attr:`_state_scalar`) and ``_state_nu``.

    Also note that if a filter represents the process or observation noise as student-t distributions,
    (or rather that it expects scale matrices instead of covariance matrices), then the filter has to
    also adapt ``self._process_scalar`` and ``self._obs_scalar``
    """

    _state_distr_class = StudentTDistribution
    _state_nu : int      = 3                      #: The degrees of freedom of the state distribution

    def _init_smoothing(self, last) -> None:
        super()._init_smoothing(last=last)
        try:
            self._state_nu = self._filter._state_nu
        except Exception:
            self._state_nu = 30+self._filter._state_dim
        self._state_scalar = utils.KLDmin_Norm_T(dim=self._filter._state_dim, nu=self._state_nu)

    def _state_distr(self) -> Dict[str, numpy.ndarray]:
        return StudentTDistribution.create_params(mu = self._state_mean, P = self._state_scalar*self._state_covar,
                                                  nu = self._state_nu)


class KalmanFilter(NormalStateFilter):
    """
    Implementation of the basic Kalman filter.
    """
    def __init__(self, model: AbstractDynamicModel, mean: numpy.ndarray, covar: numpy.ndarray, 
                 current_time: float = None, seed: int = None, exp_hist_len: int = 100, 
                 process_scalar: float = 1., obs_scalar: float = 1., gating: float = 1.0) -> None:
        """
        Initialises the Kalman Filter

        :param gating: One minus the probability to see a more unexpected observation for which the
                observation is discarded to prevent outliers from affecting the algorithm, defaults to 1
        """
        super().__init__(model=model, mean=mean, covar=covar, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._gating = gating
        self._gating_thres = scipy.stats.chi2(self._obs_dim).ppf(gating)

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        res = obs - HFm
        PtH = FPFH + QH
        K = PtH @ numpy.linalg.solve(HFPFH + HQH + R, numpy.eye(self._obs_dim))
        if self._gating > 1 -1e-10:
            self._state_mean  = Fm + K @ res
            self._state_covar = FPF+Q - K @ numpy.transpose(PtH, (0,2,1))
        else:
            gate = (numpy.transpose(res, (0,2,1)) @ numpy.linalg.solve(HFPFH+HQH+R, res) > self._gating_thres).reshape((-1,1,1))
            self._state_mean  = gate * ( Fm  ) + (not gate) * (Fm + K @ res)
            self._state_covar = gate * (FPF+Q) + (not gate) * (FPF+Q - K @ numpy.transpose(PtH, (0,2,1)))
        return {}

    @staticmethod
    def label() -> str:
        return "KF"

    def label_long(self) -> str:
        return "Kalman Filter"

    def desc(self) -> str:
        return "Kalman Filter"

    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        return "KF"

class KalmanSmoother(NormalStateSmoother):
    """
    A simple implementation of the `Rauch-Tung-Striebel smoother <https://arc.aiaa.org/doi/abs/10.2514/3.3166>`_
    """

    def _smoothen(self, mu, P, Fm, FPF, Q, PF, pre_args) -> Dict[str, numpy.ndarray]:
        k = mu.shape[0]
        for j in range(k):
            P_pri = FPF[j]+Q[j]
            P_pri_inv = scipy.linalg.solve(P_pri, numpy.eye(Q.shape[1]), check_finite=False, assume_a='pos')
            J = PF[j] @ P_pri_inv
            self._state_mean[j]  = mu[j] + J @ (self._state_mean[j]  - Fm[j])
            self._state_covar[j] = P[j]  + J @ (self._state_covar[j] - P_pri) @ J.T
        return {}

