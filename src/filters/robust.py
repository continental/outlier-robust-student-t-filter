"""
A collection of outlier robust Bayesian Filters for benchmarking.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import scipy
from . import abstract
from . import basic
from .. import utils

print("-   Loading File 'robust.py'")

class chang_RKF(basic.NormalStateFilter):
    """
    An implementation of a robust Kalman Filter (RKF) proposed in ‘Robust Kalman Filtering Based on
    Mahalanobis Distance as Outlier Judging Criterion’, Guobin Chang, 2014. https://doi.org/10.1007/s00190-013-0690-8.
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 current_time: float = None, alpha=0.01, seed: int = None, exp_hist_len: int = 100,
                 process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter

        :param alpha: Given an observation :math:`y` and expected observation :math:`\bar y` with 
                Mahalanobis distance :math:`\gamma = (y- \bar y)^\top (H P H^\top + \lambda R)^{-1} (y - \bar y)`,
                then the observation will be treated as an outlier if the probability of observing a greater
                Mahalanobis distance is less than ``alpha`` and :math:`\lambda \geq 1` is scaled up until the 
                probability is (nearly) exactly ``alpha``. Defaults to 0.05.
        """
        super().__init__(model=model, mean=mean, covar=covar, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._alpha = alpha
        self._thres = scipy.stats.chi2(self._obs_dim).ppf(1-alpha)
        k = self._state_mean.shape[0]
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["kappas"] = np.empty((exp_hist_len+1, k))
        self._hist["kappas"][0] = np.ones((k,), dtype=float)

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        HPH = HFPFH + HQH
        it, k, m = (1, Fm.shape[0], self._obs_dim)
        kappa = np.ones((k,), dtype=float)
        for i in range(k):
            res = obs[i] - HFm[i]
            HPHi, Ri = (HPH[i], R[i])
            #Pn = np.linalg.solve(HPHi + kappa[i]*Ri, res)
            Pn = scipy.linalg.solve(HPHi + kappa[i]*Ri, res, check_finite=False, assume_a='pos')
            gamma = (res.T @ Pn)[0,0]    # The Mahalanobis distance
            while gamma > self._thres:
                kappa[i] += (gamma - self._thres) /  (Pn.T @ Ri @ Pn)[0,0]
                #Pn = np.linalg.solve(HPHi + kappa[i]*Ri, res)
                Pn = scipy.linalg.solve(HPHi + kappa[i]*Ri, res, check_finite=False, assume_a='pos')
                gamma = (res.T @ Pn)[0,0]
                it += 1
        
            PH  = FPFH[i] + QH[i]
            #PHS = PH @ np.linalg.solve(HPHi + kappa[i]*Ri, np.eye(m))
            PHS = PH @ scipy.linalg.solve(HPHi + kappa[i]*Ri, np.eye(m), check_finite=False, assume_a='pos')
            self._state_mean[i]  = Fm[i] + PHS @ res
            self._state_covar[i] = FPF[i]+Q[i] - PHS @ PH.T
            self._state_covar[i] = (self._state_covar[i] + self._state_covar[i].T)/2

        return {"iterations": [it], "kappas":[np.squeeze(kappa)]}

    @staticmethod
    def label() -> str:
        return "RKF"
    
    def label_long(self) -> str:
        return "RKF Chang"

    def desc(self) -> str:
        return "Robust Kalman Filter after Guobin Chang, 2014. alpha = {:.2f}%".format(self._alpha*100)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc > 18 or not hardcap:
            return r"RKF Chang, $\alpha$={:.1f}%".format(self._alpha*100)
        # hardcap is True
        return "RKF Chang"[:hardcap]



class chang_ARKF(basic.NormalStateFilter):
    r"""
    An implementation of a adaptive and robust Kalman Filter (ARKF) proposed in ‘Kalman filter with
    both adaptivity and robustness’, Guobin Chang, 2014. https://doi.org/10.1016/j.jprocont.2013.12.017

    The ARKF algorithm works in the way that if an outlier is observed (``gamma > self._thres``, 
    which is equivalent to the probability to generate an observation at least as far away from the
    prediction than the measurement under the current state estimate is less than :math:`\alpha`)
    then there will be three hypothesis for the current state, the Kalman Filter (KF), the
    Robust Kalman Filter (RKF) and Adaptive Kalman Filter (AKF) hypotheses. 
    Only when observing the next measurement will be decided which hypothesis was the most likely
    correct one.
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 current_time: float = None, alpha=0.01, seed: int = None, exp_hist_len: int = 100,
                 process_scalar: float = 1., obs_scalar: float = 1., max_lamb:float = 1e+2) -> None:
        r"""
        Constructs the filter. Important difference to other filters: ``mean`` and ``covar`` the 
        initial values for the ``_state_mean`` and ``_state_covar`` variables will be a stack of 
        three different means and covariance matrices to represent the KF, RKF and AKF hypotheses.
        Of course, if only a single mean and covariance is given, they will be duplicated.

        :param alpha: Given an observation :math:`y` and expected observation :math:`\bar y` with 
                Mahalanobis distance :math:`\gamma = (y- \bar y)^\top (H P_{t+1|t} H^\top + \lambda R)^{-1} (y - \bar y)`,
                then the observation will be treated as an outlier if the probability of observing a greater
                Mahalanobis distance is less than ``alpha`` and :math:`\lambda \geq 1` is scaled up until the 
                probability is (nearly) exactly ``alpha``. Defaults to 0.01.
        :param max_lamb: The maximal scalar value for the AKF parameter lambda. Defaults to 100 
                (larger values easily lead to instabilities, especially if multiple sensors are considered).
        """
        mu = np.atleast_1d(np.squeeze(mean))
        mu = mu[:,:,None] if mu.ndim == 2 else mu[None,:,None]
        assert mu.shape[0] in [1,3], f"Need exactly three or one initial means but mean has shape, {mean.shape}"
        mean = mu if mu.shape[0] == 3 else np.repeat(mu,3, axis=0)

        covar = np.atleast_2d(covar)
        if covar.ndim == 2:
            covar = covar[None, :, :]
            covar = np.concatenate((covar, covar, covar), axis=0)
        elif (covar.ndim == 3 and covar.shape[0]==1):
            covar = np.concatenate((covar, covar, covar), axis=0)
        super().__init__(model=model, mean=mean, covar=covar, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._alpha = alpha
        self._thres = scipy.stats.chi2(self._obs_dim).ppf(1-alpha)
        self.max_lamb = max_lamb


        # Also note that the parameters for 'methods' and the distribution parameters in 'estimates'
        # can only be updated one step into the future, therefore these have to be specially rectified
        # in a postprocessing step
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["kappas"] = np.empty((exp_hist_len+1, 1))
        self._hist["kappas"][0] = 0
        self._hist["lambdas"] = np.empty((exp_hist_len+1, 1))
        self._hist["lambdas"][0] = 0
        self._pos_outlier = False
        self._methods = ["KF", "RKF", "AKF"]
        self._hist["methods"] = np.empty((exp_hist_len+1, 1), dtype=int)
        self._hist["methods"][0] = 0
        

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        HPH = HFPFH + HQH
        res = obs - HFm
        
        # Given the new observation, test which hypothesis was the most likely correct one.
        # For this, we need to update the self._state_mean and self._state_covar variables, such
        # that _state_distr() can return the correct state-distribution-estimate - Of the last step!
        if not self._pos_outlier:
            # No outlier was observed in the last step
            method = 0
        else:
            best_crit = -np.inf
            for i in range(3):
                HPH = HFPFH[i] + HQH[i]
                res = obs[0] - HFm[i]
                #nP  = np.linalg.solve(HPH + R[i], res)
                nP  = scipy.linalg.solve(HPH + R[i], res, check_finite=False, assume_a='pos')
                fac = -self._obs_dim/2*np.log(2*np.math.pi)-1/2*np.log(np.linalg.det(HPH+R[i]))
                #fac = 0
                gamma =  np.squeeze(res.T @ nP)
                # Instead of choosing the one with the best gamma, we choose the hypothesis with the
                # largest likelihood of producing the observation. 
                # Otherwise extremely large P (i.e. with high uncertainty) will always be prefered,
                # since they have small mahalanobis distance
                if fac-1/2*gamma > best_crit:
                    best_idx = i
                    best_crit = fac-1/2*gamma
            method = best_idx
        
        it, n, m = (1, self._state_dim, self._obs_dim)
        lamb, kappa = (1.,1.)
        HPH    = HFPFH[method] + HQH[method]
        res    = obs[0] - HFm[method]
        #nP     = np.linalg.solve(HPH + R[method], res)
        nP     = scipy.linalg.solve(HPH + R[method], res, check_finite=False, assume_a='pos')
        gamma  = np.squeeze(res.T @ nP)

        # Kalman Filter
        PH = FPFH[method] + QH[method]
        #K  = PH @ np.linalg.solve(HPH + R[method], np.eye(m))
        K  = PH @ scipy.linalg.solve(HPH + R[method], np.eye(m), check_finite=False, assume_a='pos')
        mu_KF = Fm[method] + K @ res
        P_KF  = FPF[method] + Q[method] - K @ PH.T
        P_KF  = (P_KF + P_KF.T)/2

        # if observation is explained by the standard modell sufficiently well, use Kalman updates
        if gamma <= self._thres:
            self._pos_outlier = False

            self._state_mean  = np.concatenate( (mu_KF[None,:,:], np.zeros((2,n,1))), axis=0 )
            self._state_covar = np.stack( (P_KF, np.eye(n), np.eye(n)), axis=0  )
        else:
            self._pos_outlier = True
            # Robust Kalman Filter
            nP_RKF, gamma_RKF = (nP, gamma)
            while gamma_RKF > self._thres:
                add_kappa = (gamma_RKF - self._thres) / np.squeeze(nP_RKF.T @ R[method] @ nP_RKF)
                kappa += add_kappa
                #nP_RKF = np.linalg.solve(HPH + kappa*R[method], res)
                nP_RKF = scipy.linalg.solve(HPH + kappa*R[method], res, check_finite=False, assume_a='pos')
                gamma_RKF = np.squeeze(res.T @ nP_RKF)
                it += 1
                if add_kappa/kappa < 1e-4:
                    break
            #K  = PH @ np.linalg.solve(HPH + kappa*R[method], np.eye(m))
            K  = PH @ scipy.linalg.solve(HPH + kappa*R[method], np.eye(m), check_finite=False, assume_a='pos')
            mu_RKF = Fm[method] + K @ res
            P_RKF  = FPF[method] + Q[method] - K @ PH.T
            P_RKF  = (P_RKF + P_RKF.T)/2

            # Adaptive Kalman Filter
            nP_AKF, gamma_AKF = (nP, gamma)
            while gamma_AKF > self._thres:
                add_lamb = (gamma_AKF - self._thres) /  np.squeeze(nP_AKF.T @ HFPFH[method] @ nP_AKF)
                lamb += add_lamb
                try:
                    #nP_AKF = np.linalg.solve(lamb*HFPFH[method] + HQH[method] + R[method], res)
                    nP_AKF = scipy.linalg.solve(lamb*HFPFH[method] + HQH[method] + R[method], res, check_finite=False, assume_a='pos')
                except Exception as e:
                    mat = lamb*HFPFH[method] + HQH[method] + R[method]
                    print("\nProblem in ARKF AKF step!")
                    print(f"proposed lambda {lamb} leads to matrix\n",
                          utils.nd_to_str(np.squeeze(mat)))
                    print("which has eigenvalues: ", utils.nd_to_str(np.linalg.eig(np.squeeze(mat))[0]))
                    raise e
                gamma_AKF = np.squeeze(res.T @ nP_AKF)
                it += 1
                if add_lamb/lamb < 1e-4:
                    break
                if lamb > self.max_lamb:
                    lamb = self.max_lamb
                    break
            PH = lamb*FPFH[method] + QH[method]
            #K  = PH @ np.linalg.solve(lamb*HFPFH[method] + HQH[method] + R[method], np.eye(m))
            K  = PH @ scipy.linalg.solve(lamb*HFPFH[method] + HQH[method] + R[method], np.eye(m), check_finite=False, assume_a='pos')
            mu_AKF = Fm[method] + K @ res
            P_AKF  = lamb*FPF[method] + Q[method] - K @ PH.T
            P_AKF  = (P_AKF + P_AKF.T)/2
            if np.any(np.diag(P_AKF) < 0):
                print("Negative AKF results detected: lambda = ", lamb)
                print("\nP:\n",utils.nd_to_str(lamb*FPF[method] + Q[method]))
                print("PH:\n",utils.nd_to_str(lamb*FPFH[method] + QH[method]))
                print("S:\n", utils.nd_to_str(scipy.linalg.solve(lamb*HFPFH[method] + HQH[method] + R[method], np.eye(m), check_finite=False, assume_a='pos')))
                print("K:\n", utils.nd_to_str(K))
                print("KPH:\n", utils.nd_to_str(K @ PH.T))

            self._state_mean  = np.stack( (mu_KF, mu_RKF, mu_AKF), axis=0 )
            self._state_covar = np.stack( (P_KF,  P_RKF,  P_AKF),  axis=0  )

        # Note that we have to track the method used in the last step, this will be fixed in the
        # postprocessing step
        return {"iterations": [it], "kappas": [kappa], "lambdas": [lamb], "methods": [method]}

    def _filter_postprocess(self, processed_steps : int = 1):
        # The last newly filled entry in _hist
        last = self._hist_idx-1
        # The first index that has to be updated 
        # (since the hypotheses of the last filtering step from the last filter() call has only been
        #  studied in the current filter() call, it also has to be updated. However, only if this is
        #  not the very first time filter() was called.)
        first = max(0, self._hist_idx-processed_steps-1)
        self._hist["methods"][first:last] = self._hist["methods"][first+1:last+1]
        # Similarly, we have not yet decided which hypothesis is the most likely one for the very last
        # step, therefore we will choose the KF hypothesis if we do not suspect an outlier and RKF 
        # otherwise
        self._hist["methods"][last] = int(self._pos_outlier)
        
    def get_state_distr(self, start_time: float | int | None = None, end_time: float | int | None = None, 
                        indexed=False) -> tuple(abstract.AbstractDistribution, np.ndarray):
        r"""
        Overwrites the parent :meth:`AbstractFilter.get_state_distr` since we have to only return
        the chosen hypotheses.
        """
        if start_time is None:
            first_idx = self._hist_idx-1
        elif np.isneginf(start_time):
            first_idx = 0
        else:
            if indexed:
                first_idx = min( max(int(start_time),0), self._hist_idx-1)
            else:
                first_idx = np.searchsorted(self._hist["times"][:self._hist_idx, 0], start_time, side="left")
        if end_time is None:
            last_idx = first_idx +1
        elif np.isposinf(end_time):
            last_idx = self._hist_idx
        else:
            if indexed:
                last_idx = min( max(int(end_time),  0), self._hist_idx)
            else:
                last_idx = np.searchsorted(self._hist["times"][:self._hist_idx, 0], end_time, side="right")
        params = {name: arr[first_idx:last_idx] for name, arr in self._hist["estimates"].items()}
        idxs = self._hist["methods"][first_idx:last_idx].reshape((-1, 1, 1, 1))
        params["mu"]  = np.take_along_axis(params["mu"],  idxs, axis=1)
        params["P"]   = np.take_along_axis(params["P"],   idxs, axis=1)
        params["cho"] = np.take_along_axis(params["cho"], idxs, axis=1)
        params["fac"] = np.take_along_axis(params["fac"], idxs.reshape((-1, 1, 1)), axis=1)
        return ( self._state_distr_class(params = params),
                 self._hist["times"][first_idx:last_idx,0] )

    @staticmethod
    def label() -> str:
        return "ARKF"
    
    def label_long(self) -> str:
        return "ARKF Chang"

    def desc(self) -> str:
        return "Adaptive and Robust Kalman Filter after Guobin Chang, 2014. alpha = {:.2f}%".format(self._alpha*100)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc > 19 or not hardcap:
            return r"ARKF Chang, $\alpha$={:.1f}%".format(self._alpha*100)
        # hardcap is True
        return "ARKF Chang"[:hardcap]



class roth_STF(basic.StudentTStateFilter):
    """
    An implementation of the robust Filter based on student-t distributions proposed by Roth in 
    "Kalman Filters for Nonlinear Systems and Heavy-Tailed Noise", Michael Roth, 2013
    
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 current_time: float = None, state_nu=None, process_gamma=None, obs_delta=3, seed: int = None, 
                 exp_hist_len: int = 100, process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter
        """
        process_gamma = process_gamma if process_gamma is not None else obs_delta+self._obs_dim+self._state_dim
        state_nu = state_nu if state_nu is not None else obs_delta+self._obs_dim
        super().__init__(model=model, mean=mean, covar=covar, state_nu=state_nu, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._process_gamma = process_gamma
        self._obs_delta = obs_delta
        self._process_scalar = utils.KLDmin_Norm_T(dim=self._state_dim, nu=process_gamma)
        self._obs_scalar = utils.KLDmin_Norm_T(dim=self._obs_dim, nu=obs_delta)


    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        k, m, nu = (Fm.shape[0], self._obs_dim, self._state_nu)
        gamma, delta = (self._process_gamma, self._obs_delta)
        # Approximating the joint distribution of :math:`x \sim t_{\nu}(\mu, \Sigma)` and
        # :math:`v \sim t_{\gamma}(0, Q)` by a joint distribution with degrees of freedom 
        # :math:`\min(\nu, \gamma)` and covariance matrix :math:`diag(p_f \Sigma, q_f Q)`.
        # -> currently no Kullback-Leibler minimisation between two Student-t distribution, thus we
        # use simple moment matching
        nu_tilde = min(nu, gamma)
        nu_bar   = min(nu_tilde, delta)
        pf       = nu*(nu_tilde-2)/(nu-2)/nu_tilde
        qf       = gamma*(nu_tilde-2)/(gamma-2)/nu_tilde
        pf2      = nu_tilde*(nu_bar-2)/(nu_tilde-2)/nu_bar
        rf       = delta*(nu_bar-2)/(delta-2)/nu_bar
        
        for i in range(k):
            res = obs[i] - HFm[i]
            P   = pf2*(pf* FPF[ i] + qf* Q[ i])
            PH  = pf2*(pf* FPFH[i] + qf* QH[i])
            HPH = pf2*(pf*HFPFH[i] + qf*HQH[i])
            #S   = np.linalg.solve(HPH + rf * R[i], np.eye(m))
            S   = scipy.linalg.solve(HPH + rf * R[i], np.eye(m), check_finite=False, assume_a='pos')
            D   = res.T @ S @ res
            K   = PH @ S
            self._state_mean[i]  = Fm + K @ res
            self._state_covar[i] = (nu_bar + D)/(nu_bar + m)* ( P - K @ PH.T )
            self._state_covar[i] = (self._state_covar[i] + self._state_covar[i].T)/2
        self._state_nu = nu_bar + m
        self._state_scalar = utils.KLDmin_Norm_T(dim=self._state_dim, nu=self._state_nu)

        return {}

    @staticmethod
    def label() -> str:
        return "STF Roth"
    
    def label_long(self) -> str:
        return "STF Roth"

    def desc(self) -> str:
        return "Robust Stundent-T Filter after Michael Roth, 2013. nu={:}, gamma={:}, delta={:}".format(self._state_nu, self._process_gamma, self._obs_delta)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc > 24 or not hardcap:
            return "STF Roth, $\\nu={:}$, $\\gamma={:}$, $\\delta={:}$".format(self._state_nu, self._process_gamma, self._obs_delta)
        return "STF Roth"[:hardcap]



class Agamennoni_VBF(basic.NormalStateFilter):
    """
    An implementation of the robust Variational Bayes Filter proposed by Gabriel Agamennoni, Juan 
    Nieto and Eduardo Nebot in "An outlier-robust Kalman filter" 2011. https://doi.org/10.1109/ICRA.2011.5979605
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 current_time: float = None, nu : float = 8, gating : float = 1, seed: int = None, 
                 exp_hist_len: int = 100, process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter

        :params nu: the degrees of freedom of the Wishard distribution, has to be greater than 1.
        :type nu: float
        :params gating: The probability with which an observation is not gated. I.e. Observations 
                outside of the gating-elipsoid will be ignored.
        :type gating: float or None
        """
        super().__init__(model=model, mean=mean, covar=covar, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._nu = nu
        self._gating = gating
        self._gating_thres = scipy.stats.chi2(self._obs_dim).ppf(gating)
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        k, n, m, nu = (Fm.shape[0], self._state_dim, self._obs_dim, self._nu)
        
        for i in range(k):
            Gl   = R[i] # last Gamma
            it = 1
            res  = obs[0] - HFm[i]
            #H    = self._model._H
            _P   =  FPF[i]  +  Q[i]
            _PH  =  FPFH[i] +  QH[i]
            HPH  = HFPFH[i] + HQH[i]
            #gamma = res.T @ np.linalg.solve(HPH + Gl, res)
            gamma = res.T @ scipy.linalg.solve(HPH + Gl, res, check_finite=False, assume_a='pos')
            # gating
            if gamma < self._gating_thres:
                norm_Gl = np.sqrt((Gl**2).sum())
                while True:
                   #___S    = np.linalg.solve(Gl + HPH, np.eye(m))
                    ___S    = scipy.linalg.solve(Gl + HPH, np.eye(m), check_finite=False, assume_a='pos')
                    _PHS    = _PH  @ ___S
                   #_PHSHP  = _PHS @ _PH.T
                   #_PHSHPH = _PHS @ HPH
                    HPHS    = HPH  @ ___S
                   #HPHSHP  = HPHS @ HP
                   #HPHSHPH = HPHS @ HPH

                    # In the Agamennoni et al. paper, they use the different notation
                    #   F = A, H = C

                    # We use a slightly different form of Joseph's form here
                    # 
                    # For the Iteration we need: H Sigma H.T where by Josephs form
                    #       Sigma = P H.T S Gl S H P + (In - P H.T S H) P (In - H.T S H P)
                    # note how we can reformulate 
                    #       H Sigma H.T = H P H.T S Gl S H P H.T + (Im - H P H.T S ) H P H.T (Im - S H P H.T)
                    # If we now look at Im - HPH S we note with S = (HPH + Gl)^-1 that this equals
                    #       Im - HPH (HPH + Gl)^-1 = Im - (HPH + Gl)(HPH + Gl)^-1 + Gl (HPH + Gl)^-1
                    #     = GL S
                    GlS = Gl @ ___S
                    newHPH = HPHS @ Gl @ HPHS.T + GlS @ HPH @ GlS.T
                    newHPH = (newHPH + newHPH.T)/2
                    
                    # d = obs - Hμ = obs - HFm - HPHS @ res = res - HPHS @ res
                    d   = res - HPHS @ res
                    G   =  ( (nu-1)* R[i] + d @ d.T + newHPH) / nu
                    G   = (G + G.T)/2
                    norm_G = np.sqrt((G**2).sum())
                    error  = ((G-Gl)**2).sum()**0.5 / (norm_G + norm_Gl)
                    Gl = G
                    norm_Gl = norm_G
                    if  error  <  1e-3:
                        self._state_mean[i] = Fm[i] + _PHS @ res
                        # since after convergence of G the following should be stable 
                        self._state_covar[i] = _P - _PHS @ _PH.T
                        break
                    it += 1
            else:
                self._state_mean[i] = Fm[i]
                self._state_covar[i] = _P
                
            #print("new P-P.T:\n",utils.nd_to_str(si[0]-si[0].T, precision=20))

        # for further stabilisation. TODO find where the instability lies, the old implementation
        # seems to struggle less
        #print("max Transpose error: ", np.max(np.abs(self._state_covar-np.transpose(self._state_covar, (0,2,1)))))
        self._state_covar = (self._state_covar+np.transpose(self._state_covar, (0,2,1)))/2

        return {"iterations": [it]}

    @staticmethod
    def label() -> str:
        return "VBF Agam."

    def label_long(self) -> str:
        return "VBF Agamennoni"

    def desc(self) -> str:
        return ("Variational Bayes Filter based on Wishart distributions after Agamennoni et al.,"
                +f" 2011. $\nu = {self._nu:}$"+(f", gated after {int(100*self._gating):.1f}%" if self._gating < 1 else ""))
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc > 24 or not hardcap:
            return f"VBF Agamennoni, $\nu={self._nu:}$" +(f", G {int(100*self._gating):}%" if self._gating < 1 else "")
        return (f"VBF Agamennoni, $\nu={self._nu}$"+(f" gated {int(100*self._gating):}" if self._gating < 1 else "") )[:hardcap]




class Saerkkae_VBF(basic.NormalStateFilter):
    """
    An implementation of the robust Variational Bayes Filter proposed by Simo Särkkä and Aapo 
    Nummenmaa in "Recursive Noise Adaptive Kalman Filtering by Variational Bayesian Approximations",
    2009. https://doi.org/10.1109/TAC.2008.2008348

    To explain outliers they assume that the process noise has a covariance matrix :math:`R`, which
    in itself is a diagonal matrix with inversly Gamma-distributed entries.
    Note that this is similar to the Wishart assumtion of Agamennoni et al, since the chi-squared 
    distribution (which is used inversely to scale the matrices) is a special case of the Gamma distribution.
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 current_time: float = None, rho : float = 0.5, alpha : np.ndarray | float = 1/2, 
                 beta : np.ndarray | float = 1, gating : float = 1, seed: int = None, 
                 exp_hist_len: int = 100, process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter

        :params rho: Indicator of the memory-capacity for the observation noise covariance :math:`R_k`.
                They assume that the diagonal entries :math:`(R_k)_{ii}` are inversely Gamma distributed
                :math:`\frac{1}{(R_k)_{ii}} \sim \Gamma(\alpha_{k,i}, \beta_{k,i})`. And these
                :math:`\alpha, \beta` parameters depend on their previous version by 
                :math:`\alpha_{k,i}|\alpha_{k-1,i} = \frac{1}{2}+\rho\alpha_{k-1,1}` and 
                :math:`\beta_{k,i}|\beta_{k-1,i} = \rho\beta_{k-1,1}`
        :type rho: float
        :params alpha: The initial estimations for the :math:`\alpha_{0,i}`
        :params beta: The initial estimations for the :math:`\beta_{0,i}`
        :params gating: The probability with which an observation is not gated. I.e. Observations 
                outside of the gating-elipsoid will be ignored.
        :type gating: float or None
        """
        super().__init__(model=model, mean=mean, covar=covar, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._rho   = rho
        k = self._state_mean.shape[0]
        self._alpha = np.ones((k,self._obs_dim)) * beta
        self._beta  = np.ones((k,self._obs_dim)) * alpha
        self._gating = gating
        self._gating_thres = scipy.stats.chi2(self._obs_dim).ppf(gating)
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["alphas"] = np.empty((exp_hist_len+1, k, self._obs_dim))
        self._hist["alphas"][0] = self._alpha
        self._hist["betas"] = np.empty((exp_hist_len+1, k, self._obs_dim))
        self._hist["betas"][0] = self._beta

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        k, m = (Fm.shape[0], self._obs_dim)
        mu = np.empty(Fm.shape)
        si = np.empty(FPF.shape)
        
        for i in range(k):
            res  = obs[0] - HFm[i]
            _P   =  FPF[i]  +  Q[i]
            _PH  =  FPFH[i] +  QH[i]
            HPH  = HFPFH[i] + HQH[i]
            a    = 1/2 + self._rho * self._alpha[i]
            # The (1-rho) part is added for cases with rho = 0 and so that the model noise variance
            #  is also considered, ie for a better initial guess of beta 
            b    = self._rho * self._beta[i] + (1-self._rho)*np.diag(R[i])
            bn   = b
            G    = np.diag(bn/a)
            #gamma = res.T @ np.linalg.solve(HPH + G, res)
            gamma = res.T @ scipy.linalg.solve(HPH + G, res, check_finite=False, assume_a='pos')
            # gating
            if gamma < self._gating_thres:
                it = 1
                while True:
                    #S  = np.linalg.solve(HPH + G, np.eye(m))
                    S  = scipy.linalg.solve(HPH + G, np.eye(m), check_finite=False, assume_a='pos')
                    ino = HPH @(S @res)
                    # obs - H(Fm + PHS res) = res - HPHS res
                    # HPH = HPH - HPH S HPH  ( = HPH - HPH (HPH + G)^-1 HPH = HPH (HPH + G)^-1 G )
                    bnew = b + (res-ino)[:,0]**2/2 + np.diag(HPH - HPH @ S @ HPH)/2
                    error = ((bn-bnew)**2).sum()**0.5 / ((bn**2).sum()**0.5 + (bnew**2).sum()**0.5)
                    if  error  <  1e-3:
                        K = _PH @ S
                        mu[i] = Fm[i] + K @ res
                        si[i] = _P - K @ _PH.T
                        self._alpha[i] = a
                        self._beta[i] = bn
                        #print(f"k {k}: beta = {utils.nd_to_str(np.squeeze(bnew))}, mu = {utils.nd_to_str(np.squeeze(mu[i]))}, P[0,0] = {utils.nd_to_str(np.squeeze(si[i,0,0]))}")
                        break
                    bn = bnew
                    G    = np.diag(bn/a)
                    it += 1
            else:
                mu[i] = Fm[i]
                si[i] = _P

        self._state_mean  = mu
        self._state_covar = si

        return {"iterations": [it], "alphas": self._alpha[None,:,:], "betas": self._beta[None,:,:]}

    @staticmethod
    def label() -> str:
        return "VBF Särk."
    
    def label_long(self) -> str:
        return "VBF Särkkä"

    def desc(self) -> str:
        return ("Variational Bayes Filter based on Inverse Gamma distributions after Särkkä et al.,"
                +f" 2009. $\\rho = {self._rho:}$"+(f", gated after {int(100*self._gating):.1f}%" if self._gating < 1 else ""))
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc > 24 or not hardcap:
            return f"VBF Särkkä, $\\rho ={self._rho:.2f}$"+(f", gated after {int(100*self._gating):.1f}%" if self._gating < 1 else "")
        return (f"VBF Särkkä, $\\rho ={self._rho:.2f}$"+(f" gated {int(100*self._gating):}" if self._gating < 1 else "") )[:hardcap]



class Huang_SSM(basic.NormalStateFilter):
    """
    An implementation of the robust Filter based on Statistical Similarity Measures proposed by 
    Yulong Huang, Yonggang Zhang, Yuxin Zhao, Peng Shi, and Jonathon Chambers in 
    "A Novel Outlier-Robust Kalman Filtering Framework Based on Statistical Similarity Measure" 2021.
    https://doi.org/10.1109/TAC.2020.3011443
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 current_time: float = None, SSM : str = "log", nu_SSM : float = 3, sigma_SSM : float = 1, 
                 gating : float = 1, process_non_gaussian : bool = True, noise_non_gaussian : bool = True, 
                 separate : bool = True, delta : float = 1e-8, seed: int = None,
                 exp_hist_len: int = 100, process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter

        :param SSM: Which Similarity meassure should be choosen, the options are
                #. ``log`` for :math:`f(\Delta) = -\frac{\nu + d}{2} \cdot \log\left(1+\frac{\Delta}{\nu}\right)`
                #. ``exp`` for :math:`f(\Delta) = \sigma \cdot e^{\frac{d-\Delta}{2 \sigma}}`
                #. ``sqrt`` for :math:`f(\Delta) = -\sqrt{(\nu+d)\cdot (\nu+\Delta)}`
                #. ``sq`` for :math:`f(\Delta) = -\frac{\Delta^2}{4}`
                #. ``lin`` for :math:`f(\Delta) = -\frac{\Delta}{2}`
                defaults to ``sqrt``.
        :param nu_SSM: the parameter :math:`\nu` for some similarity measures, see above. Defaults to 3.
        :param sigma_SSM: the parameter :math:`\sigma` for some similarity measures, see above. Defaults to 1
        :params gating: The probability with which an observation is not gated. I.e. Observations 
                outside of the gating-elipsoid will be ignored.
        :param separate: Whether the separate algorithm should be used - i.e. in which first 
                :math:`\lambda` and then :math:`\xi` is calculated, that is, if the other is assumed
                to be constant. Defaults to True
        :param process_non_gaussian: Whether outliers (i.e. jumps) in the process noise are feasible.
                Defaults to False
        :param noise_non_gaussian: Whether outliers in the observation noise are feasible.
                Defaults to True
        :param delta: The smallest possible values for :math:`\lambda` and then :math:`\xi`, defaults to :math:`10^{-8}`.
        """
        super().__init__(model=model, mean=mean, covar=covar, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._SSM = SSM
        k, n, m = (self._state_mean.shape[0], self._state_dim, self._obs_dim)
        if SSM == 'log':
           #self._fx,  self._fy  = (lambda t: -(nu + n_process)/2 * np.log(1+t/nu),     lambda t: -(nu + n_noise  )/2 * np.log(1+t/nu) )
            self._dfx, self._dfy = (lambda t: -(nu_SSM + n)/(nu_SSM + t)/2,             lambda t: -(nu_SSM + m  )/(nu_SSM + t)/2 )
        elif SSM == 'exp':
            sigma_SSM = sigma_SSM**2
           #self._fx,  self._fy  = (lambda t: sigma_SSM * np.exp((n-t)/2/sigma_SSM),    lambda t: sigma_SSM * np.exp((m  -t)/2/sigma_SSM) )
            self._dfx, self._dfy = (lambda t: -np.exp((n-t)/2/sigma_SSM)/2,             lambda t: -np.exp((m  -t)/2/sigma_SSM)/2 )
        elif SSM == 'sqrt':
           #self._fx,  self._fy  = (lambda t: -np.sqrt( (nu_SSM+n)*(nu_SSM+t) ),        lambda t: -np.sqrt( (nu_SSM+m  )*(nu_SSM+t) ) )
            self._dfx, self._dfy = (lambda t: -np.sqrt( (nu_SSM+n)/(nu_SSM+t) )/2,      lambda t: -np.sqrt( (nu_SSM+m  )/(nu_SSM+t) )/2 )
        elif SSM == 'sq':
           #self._fx,  self._fy  = (lambda t: -t**2/4,      lambda t: -t**2/4)
            self._dfx, self._dfy = (lambda t: -t/2,         lambda t: -t/2)
        elif SSM == 'lin':
           #self._fx,  self._fy  = (lambda t: -t/2,         lambda t: -t/2)
            self._dfx, self._dfy = (lambda t: -1/2 + 0*t,   lambda t: -1/2 + 0*t) # semms ridiculous but usefull to keep the shapes of the outputs correct
        self._gating = gating
        self._gating_thres = scipy.stats.chi2(self._obs_dim).ppf(gating)
        self._process_non_gaussian = process_non_gaussian
        self._noise_non_gaussian   = noise_non_gaussian
        self._separate = separate
        self._delta    = delta
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["lambdas"] = np.empty((exp_hist_len+1, k))
        self._hist["lambdas"][0] = np.ones((k,), dtype=float)
        self._hist["xis"] = np.empty((exp_hist_len+1, k))
        self._hist["xis"][0] = np.ones((k,), dtype=float)

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        k, n, m = (Fm.shape[0], self._state_dim, self._obs_dim)

        xis = np.ones((k,))
        lambs = np.ones((k,))
        d = self._delta
        
        for i in range(k):
            xi, lamb = (1.,1.)
            res  = obs[0] - HFm[i]
            _P   =  FPF[i]  +  Q[i]
            _PH  =  FPFH[i] +  QH[i]
            HPH  = HFPFH[i] + HQH[i]
            _R = R[i]
            #_Rinv = np.linalg.solve(_R, np.eye(m))
            _Rinv = scipy.linalg.solve(_R, np.eye(m), check_finite=False, assume_a='pos')
            #P_inv = np.linalg.solve(_P, np.eye(n))
            P_inv = scipy.linalg.solve(_P, np.eye(n), check_finite=False, assume_a='pos')
            #gamma = res.T @ np.linalg.solve(HPH + _R, res)
            gamma = res.T @ scipy.linalg.solve(HPH + _R, res, check_finite=False, assume_a='pos')
            # gating
            if gamma < self._gating_thres:
                it = 1
                if self._separate:
                    if self._noise_non_gaussian:
                        for _ in range(30):
                            #S = np.linalg.solve(HPH/xi + _R/lamb, np.eye(m))
                            S = scipy.linalg.solve(HPH/xi + _R/lamb, np.eye(m), check_finite=False, assume_a='pos')
                            HK = HPH/xi @ S
                            res_new = res - HK @ res
                            B = res_new @ res_new.T + HPH/xi - HK @ HPH.T/xi
                            # remember that trace(A @ B) = \sum_ij A_ij * B_ij
                            lamb_new = -2 * self._dfy( np.sum(B*_Rinv) )
                            lamb_new = lamb_new if lamb_new > d else d
                            if abs(lamb_new-lamb)/lamb < 1e-6:
                                break
                            lamb = lamb_new
                            it += 1
                    if self._process_non_gaussian:
                        for _ in range(30):
                            #S = np.linalg.solve(HPH/xi + _R/lamb, np.eye(m))
                            S = scipy.linalg.solve(HPH/xi + _R/lamb, np.eye(m), check_finite=False, assume_a='pos')
                            K = _PH/xi @ S
                            ino = K @ res
                            A = ino @ ino.T + _P/xi - K @ _PH.T/xi
                            xi_new = -2 * self._dfx( np.sum(A*P_inv) )
                            xi_new = xi_new if xi_new > d else d
                            if abs(xi_new-xi)/xi < 1e-6:
                                break
                            xi = xi_new
                            it += 1
                else:
                    for _ in range(60):
                        #S = np.linalg.solve(HPH/xi + _R/lamb, np.eye(m))
                        S = scipy.linalg.solve(HPH/xi + _R/lamb, np.eye(m), check_finite=False, assume_a='pos')
                        K = _PH/xi @ S
                        ino = K @ res
                        if (ino**2).sum()**0.5 / ((Fm[i]+ino)**2).sum()**0.5 < 1e-6:
                            break
                        if self._process_non_gaussian:
                            A = ino @ ino.T + _P/xi - _PH/xi @ S @ _PH.T/xi
                            xi = -2 * self._dfx( np.sum(A*P_inv) )
                            xi = xi if xi > d else d
                        if self._noise_non_gaussian:
                            res_new = res - HPH/xi @ (S @ res)
                            B = res_new @ res_new.T + HPH/xi - HPH/xi @ S @ HPH.T/xi
                            lamb = -2 * self._dfy( np.sum(B*_Rinv) )
                            lamb = lamb if lamb > d else d
                        it += 1
                #S = np.linalg.solve(HPH/xi + _R/lamb, np.eye(m))
                S = scipy.linalg.solve(HPH/xi +_R/lamb, np.eye(m), check_finite=False, assume_a='pos')
                K = _PH/xi @ S
                self._state_mean[i] = Fm[i] + K @ res
                self._state_covar[i] = _P/xi - K @ _PH.T/xi
                self._state_covar[i] = (self._state_covar[i] + self._state_covar[i].T)/2
                xis[i] = xi
                lambs[i] = lamb
            else:
                self._state_mean[i] = Fm[i]
                self._state_covar[i] = (_P+_P.T)/2


        return {"iterations": [it], "lambdas":lambs, "xis": xis}

    @staticmethod
    def label() -> str:
        return "SSMKF"

    def label_long(self) -> str:
        return f"SSMKF-{self._SSM:}"+("-S" if self._separate else "")+" Huang"

    def desc(self) -> str:
        return ("Stochastic Similarity Measurement Filter after Huang et al., 2021. "
                +f"SSMKF-{self._SSM:}"+("-S" if self._separate else "")
                +(" with no process outliers" if self._process_non_gaussian else "")
                +(" with no observation outliers" if self._noise_non_gaussian else "")
                +(f", gated after {int(100*self._gating):.1f}%" if self._gating < 1 else ""))
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc > 24 or not hardcap:
            return (f"SSMKF-{self._SSM:}"+("-S" if self._separate else "")+" Huang"
                    #+(" NormQ" if self._process_non_gaussian else "")
                    #+(" NormR" if self._noise_non_gaussian else "")
                    +(f", Thres={int(100*self._gating):.1f}%" if self._gating < 1 else "") )
        return (f"VBF Agamennoni, $\\nu={self._nu}$"+(f" gated {int(100*self._gating):}" if self._gating < 1 else "") )[:hardcap]









