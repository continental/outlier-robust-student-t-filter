# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Our proposed methods
"""

from __future__ import annotations
import numpy as np
import scipy
from typing import Dict
from . import abstract
from . import basic
from .. import utils

print("-   Loading File 'proposed.py'")

class StudentTFilter(basic.StudentTStateFilter):
    """
    An implementation of our proposed Student-t Filter
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 nu : int = 1, current_time: float = None, seed: int = None, exp_hist_len: int = 100,
                 process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter

        :param nu: The degrees of freedom :math:`\nu` of the observation distribution. That is we 
                assume that in the system dynamics

                .. math:: 
                    x_t &= F_t(x_{t-1}) + v_t, &\quad v_t &\sim t_{\nu+n+m}\,(0, Q_t)\\
                    y_t &= H_t(x_t) + e_t,     &\quad e_t &\sim t_\nu\,(0, R_t).
                
                and our state hypothesis is also a student-t distribution :math:`x_{t} \sim t_{\nu + m}\,(\mu, \Sigma)`.
                Here :math:`n` is the dimension of :math:`x_t` and :math:`m` is the dimension of 
                :math:`y_t`. Note that :math:`\Sigma` is a scale matrix and not the covariance matrix.
        """
        super().__init__(model=model, mean=mean, covar=covar, state_nu=nu +model.obs_dim, current_time=current_time, 
                         seed=seed, exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["a"] = np.empty((exp_hist_len+1, 1))
        self._hist["a"][0] = 1
        self._hist["b"] = np.empty((exp_hist_len+1, 1))
        self._hist["b"][0] = 1
        self._process_scalar = utils.KLDmin_Norm_T(dim=self._state_dim, nu=self._state_nu+self._state_dim)
        self._obs_scalar = utils.KLDmin_Norm_T(dim=self._obs_dim, nu=nu)

        # The scale matrix this algorithm works on is self._state_scalar * self._state_covar
        # instead of the normal self._state_covar
        # _state_scalar is set by the parent StudentTStateFilter class

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:

        n, m, k = (self._state_dim, self._obs_dim, Fm.shape[0])
        Id_m = np.eye(m)
        nu = self._state_nu - m
        ss = self._state_scalar


        for i in range(k):
            res = obs[i] - HFm[i]
            #a = (nu + m) / (nu + (res.T @ np.linalg.solve(R[i], res))[0,0])
            a = (nu + m) / (nu + (res.T @ scipy.linalg.solve(R[i], res, check_finite=False, assume_a='pos'))[0,0])
            b = (nu + m + n) / (nu + m + max(n-2,0) *(nu+m)/(nu+m+2))
            # we have to counteract that P (from self._state_covar) is not yet the scale matrix
            it = 0
            while it < 100:
                #S_inv = np.linalg.solve( (ss*a)*HFPFH[i] + (a*b)*HQH[i] + R[i], Id_m)
                S_inv = scipy.linalg.solve( (ss*a)*HFPFH[i] + (a*b)*HQH[i] + R[i], Id_m, check_finite=False, assume_a='pos')
                Sr = S_inv @ res
                cov_scale = (nu + (res.T @ Sr)[0,0])/(nu + m -2)
                a_approx = (nu + m) / (nu + cov_scale*(m - np.sum(R[i] * S_inv))/2 + (Sr.T @ R[i] @ Sr)[0,0])#np.einsum('ia,ij,jb->ab', r, R[i], r) )
                b_approx = (nu + m + n) / (nu + m + cov_scale*a*(n  - (ss*a)*np.sum(HFPFH[i] * S_inv))/2 + (ss*a*a) * (Sr.T @ HFPFH[i] @ Sr)[0,0])#np.einsum('ia,ij,jb->ab', r, HFPFH[i], r) )
                if abs(a_approx - a)/(a_approx+a) + abs(b_approx - b)/(b_approx+b) < 2e-2:
                    break
                a, b, it = (a_approx, b_approx, it+1)

            P_pri  = ((ss*a)*FPF[i]  + (a*b)*Q[i])
            P_priH = ((ss*a)*FPFH[i] + (a*b)*QH[i])
            K = P_priH @ S_inv

            self._state_mean[i] = Fm[i] + K @ res
            c = (nu + (res.T @ Sr)[0,0]) / (nu + m) / ss
            self._state_covar[i] = c * (P_pri - K @ P_priH.T)
            self._state_covar[i] = (self._state_covar[i] + self._state_covar[i].T)/2
        return {"iterations": [it], "a":[np.squeeze(a)], "b":[np.squeeze(b)]}

    @staticmethod
    def label() -> str:
        return "our STF"
    
    def label_long(self) -> str:
        return "our STF"

    def desc(self) -> str:
        return "proposed Student-t Filter, nu = {:}".format(self._state_nu-self._obs_dim)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc >= 17:
            return r"STF proposed, $\nu$={:}".format(self._state_nu-self._obs_dim)
        # hardcap is True
        if maxc >= 12 or not hardcap:
            return "STF proposed"
        return "STF"

class StudentTFilter_analytic(basic.StudentTStateFilter):
    """
    An initial approach to find better methods to find the scalar values :math:`a(e_k)` and :math:`b(x_{k-1})`.
    For this, we ignore computational efficiency and evaluate the distributions of :math:`a(e_k)|y_k` and :math:`b(x_{k-1})|y_k`
    as accurately as possible, and choose the median of these distributions as better approximation. 
    Surprisingly, the resulting filter performs worse, probably, since this version can actually have jumps in the process
    noise, which missfires.
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 nu : int = 1, current_time: float = None, seed: int = None, exp_hist_len: int = 100,
                 process_scalar: float = 1., obs_scalar: float = 1., N: int = 5001, M : int = 20) -> None:
        super().__init__(model=model, mean=mean, covar=covar, state_nu=nu +model.obs_dim, current_time=current_time, 
                         seed=seed, exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["a"] = np.empty((exp_hist_len+1, 1))
        self._hist["a"][0] = 1
        self._hist["b"] = np.empty((exp_hist_len+1, 1))
        self._hist["b"][0] = 1
        self._process_scalar = utils.KLDmin_Norm_T(dim=self._state_dim, nu=self._state_nu+self._state_dim)
        self._obs_scalar = utils.KLDmin_Norm_T(dim=self._obs_dim, nu=nu)
        self.N = N
        self.M = M
        
    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:

        n, m, k = (self._state_dim, self._obs_dim, Fm.shape[0])
        Id_m = np.eye(m)
        nu = self._state_nu - m
        ss = self._state_scalar

        for i in range(k):
            res = obs[i] - HFm[i]
            Rinv = scipy.linalg.solve(R[i], Id_m, check_finite=False, assume_a='pos')
            P = self._state_covar[i]
            Pinv = scipy.linalg.solve(P, np.eye(n), check_finite=False, assume_a='pos')
            PFHi = PFH[i]
            a = (nu + m) / (nu + (res.T @ (Rinv @ res))[0,0])
            b = (nu + m + n) / (nu + m + max(n-2,0) *(nu+m)/(nu+m+2))
            # we have to counteract that P (from self._state_covar) is not yet the scale matrix
            it = 0
            while it < 100:
                #S_inv = np.linalg.solve( (ss*a)*HFPFH[i] + (a*b)*HQH[i] + R[i], Id_m)
                S_inv = scipy.linalg.solve( (ss*a)*HFPFH[i] + (a*b)*HQH[i] + R[i], Id_m, check_finite=False, assume_a='pos')
                Sr = S_inv @ res
                cov_scale = (nu + (res.T @ Sr)[0,0])/(nu + m)
                a_approx = (nu + m) * utils.inv_nu_xpmTBxpm_ppf(p=0.5, m=R[i]@Sr, A=cov_scale*(R[i]-R[i]@S_inv@R[i]), B=Rinv,
                                                     nu=nu+m, s=nu, N=self.N,M=self.M)[0]
                b_approx = (nu + m + n) * utils.inv_nu_xpmTBxpm_ppf(p=0.5, m=(ss*a)*(PFHi@Sr), B=Pinv, 
                                                     A=(cov_scale*a)*(P-(ss**2 *a)*(PFHi@S_inv@PFHi.T)),
                                                     nu=nu+m+n, s=nu+m, N=self.N,M=self.M)[0]
                if abs(a_approx - a)/(a_approx+a) + abs(b_approx - b)/(b_approx+b) < 2e-3:
                    break
                a, b, it = (a_approx, b_approx, it+1)

            P_pri  = ((ss*a)*FPF[i]  + (a*b)*Q[i])
            P_priH = ((ss*a)*FPFH[i] + (a*b)*QH[i])
            K = P_priH @ S_inv

            self._state_mean[i] = Fm[i] + K @ res
            c = (nu + (res.T @ Sr)[0,0]) / (nu + m) / ss
            self._state_covar[i] = c * (P_pri - K @ P_priH.T)
            self._state_covar[i] = (self._state_covar[i] + self._state_covar[i].T)/2
        return {"iterations": [it], "a":[np.squeeze(a)], "b":[np.squeeze(b)]}

    @staticmethod
    def label() -> str:
        return "our analytic STF"
    
    def label_long(self) -> str:
        return "our analytic STF"

    def desc(self) -> str:
        return "proposed analytic Student-t Filter, nu = {:}".format(self._state_nu-self._obs_dim)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc >= 17:
            return r"STF analytic, $\nu$={:}".format(self._state_nu-self._obs_dim)
        # hardcap is True
        if maxc >= 12 or not hardcap:
            return "STF analytic"
        return "STF-ana"


class StudentTSmoother(basic.StudentTStateSmoother):
    """
    An implementation of the proposed Student-t Smoother
    """

    def _init_smoothing(self, last) -> None:
        super()._init_smoothing(last)
        self._hist["iterations"] = np.zeros((last,1), dtype=int)

    def _smoothen(self, mu, P, Fm, FPF, Q, PF, pre_args) -> Dict[str, np.ndarray]:
        n, m, k = (Fm.shape[1], self._filter._obs_dim, Fm.shape[0])
        nu = self._state_nu - m
        # Note that P and it's derivative will be attained by calling the 'cov' method of a 
        # StudentTdistribution, thus the actual scale matrix we are working on is ssMM*P with
        ssMM = (nu+m-2)/(nu+m) # MM for moment maching
        # However, the local _state_covar matrix represents a covariance matrix obtained by KLD 
        # minimisation, i.e. ss*self._state_covar is the actual scale matrix we are working with
        ss = self._state_scalar
        b_base = nu + m + max(n-2,0) *(nu+m)/(nu+m+2)
        b_init = float(pre_args["b"]) if "b" in pre_args.keys() else 1#(nu + m + n) / b_base
        for j in range(k):
            res = self._state_mean[j]  - Fm[j] # mu_{t+1} - F mu_t
            P_post = self._state_covar[j] # in usecases should be ss*P_post
            b = b_init
            db = 1
            #print(f" b: {b:.5f}")
            #while True:
            P_pri = ssMM*FPF[j]+b*Q[j]
            P_pri_inv = scipy.linalg.solve(P_pri, np.eye(Q.shape[1]), check_finite=False, assume_a='pos')
            P_pri_inv_res = P_pri_inv @ res
            J = (ssMM*PF[j]) @ P_pri_inv
            # note that we have to transform P_post from the KLD minimising covariance matrix 
            # to the scale matrix by multiplying P_post by ss
            db = ( (nu + m + (res.T @ P_pri_inv_res)[0,0] + ss*(nu+m)/(nu+m-2)*np.sum(P_pri_inv * P_post)/2 )
                        / b_base )
                # In the following a ssMM*FPF is not yet computed, but will be in it's usages
                #P_pri_inv_FPF_P_pri_inv = P_pri_inv @ FPF[j] @ P_pri_inv 
                #b_new  = (nu + m + n) / (nu + m + ssMM*(P_pri_inv_res.T @ FPF[j] @ P_pri_inv_res)[0,0] 
                #                    + 1/2*(nu + m)/(nu + m -2)*(db_new*(n-ssMM*np.sum(P_pri_inv * FPF[j])) 
                #                        + ssMM*ss*np.sum(P_pri_inv_FPF_P_pri_inv * P_post) ) )
                #b_new = b
                #print(f" b: {b_new:.5f}, db: {db:.5f}")
                #if np.abs(b_new-b)/b_new + np.abs(db-db)/db < 1e-2:
                #    break
                #b = b_new; db = db
            self._state_mean[j]  = mu[j] + J @ res
            # The following is divided by ss to transform into a KLD minimising covariance matrix
            self._state_covar[j] = ((db*ssMM/ss)*P[j]  + J @ (P_post - (db/ss)*P_pri) @ J.T)
        #print("new new covar - old covar: \n",utils.nd_to_str(np.squeeze(self._state_covar-(ssMM/ss)*P), shift=5, suppress=True))
        return {}


class StudentTFilter_GT(basic.StudentTStateFilter):
    """
    A unpractical implementation of our proposed Student-t Filter that shows the performance of the
    algorithm given that the true a and b parameters are known. However, this requires knowledge of the
    ground truth state and should only be considered for demonstration purposes.
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 nu : int = 1, current_time: float = None, seed: int = None, exp_hist_len: int = 100, 
                 process_scalar: float = 1., obs_scalar: float = 1., GTx = None, GTe = None) -> None:
        r"""
        Constructs the filter

        :param nu: The degrees of freedom :math:`\nu` of the observation distribution. That is we 
                assume that in the system dynamics

                .. math:: 
                    x_t &= F_t(x_{t-1}) + v_t, &\quad v_t &\sim t_{\nu+n+m}\,(0, Q_t)\\
                    y_t &= H_t(x_t) + e_t,     &\quad e_t &\sim t_\nu\,(0, R_t).
                
                and our state hypothesis is also a student-t distribution :math:`x_{t} \sim t_{\nu + m}\,(\mu, \Sigma)`.
                Here :math:`n` is the dimension of :math:`x_t` and :math:`m` is the dimension of 
                :math:`y_t`. Note that :math:`\Sigma` is a scale matrix and not the covariance matrix.
        :param GTx: the ground truth trajectories :math:`x_t`, need to be provided, otherwise this method won't work
        :param GTe: The ground truth observation noises :math:`e_t`, need to be provided, otherwise this method won't work
        """
        super().__init__(model=model, mean=mean, covar=covar, state_nu=nu +model.obs_dim, current_time=current_time, 
                         seed=seed, exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._hist["a"] = np.empty((exp_hist_len+1, 1))
        self._hist["a"][0] = 1
        self._hist["b"] = np.empty((exp_hist_len+1, 1))
        self._hist["b"][0] = 1
        self._process_scalar *= utils.KLDmin_Norm_T(dim=self._state_dim, nu=self._state_nu+self._state_dim)
        self._obs_scalar *= utils.KLDmin_Norm_T(dim=self._obs_dim, nu=nu)
        self.GTx = GTx
        self.GTe = GTe
        assert GTx is not None, "GTx has to be provided!"
        assert GTe is not None, "GTe has to be provided!"
        self.next_GT = 0

        # The scale matrix this algorithm works on is self._state_scalar * self._state_covar
        # instead of the normal self._state_covar
        # _state_scalar is set by the parent StudentTStateFilter class


    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:

        n, m, k = (self._state_dim, self._obs_dim, Fm.shape[0])
        nu = self._state_nu - m
        ss = self._state_scalar

        for i in range(k):
            res = obs[i] - HFm[i]
            last_x_res = self.GTx[self.next_GT] - self._state_mean[i]

            a_GT = (nu + m) / (nu + (self.GTe[self.next_GT].T @ scipy.linalg.solve(R[i], self.GTe[self.next_GT], check_finite=False, assume_a='pos'))[0,0])
            b_GT = (nu + m + n) / (nu + m + (last_x_res.T @ scipy.linalg.solve((ss)*self._state_covar[i], last_x_res, check_finite=False, assume_a='pos'))[0,0])
            
            S_inv = scipy.linalg.solve( (ss*a_GT)*HFPFH[i] + (a_GT*b_GT)*HQH[i] + R[i], np.eye(m), check_finite=False, assume_a='pos')
            P_pri  = ((ss*a_GT)*FPF[i]  + (b_GT*a_GT)*Q[i])
            P_priH = ((ss*a_GT)*FPFH[i] + (b_GT*a_GT)*QH[i])
            K = P_priH @ S_inv

            self._state_mean[i] = Fm[i] + K @ res
            c = (nu + (res.T @ S_inv @ res)[0,0]) / (nu + m) / ss
            self._state_covar[i] = c * (P_pri - K @ P_priH.T)
            
        self.next_GT += 1
        return {"a":[np.squeeze(a_GT)], "b":[np.squeeze(b_GT)]}


    @staticmethod
    def label() -> str:
        return "our STF, oracle $a_k, b_k$"
    
    def label_long(self) -> str:
        return "our STF, GT a,b"

    def desc(self) -> str:
        return "proposed Student-t Filter given ground truth knowledge, nu = {:}".format(self._state_nu-self._obs_dim)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc >= 17:
            return r"STF proposed given a,b, $\nu$={:}".format(self._state_nu-self._obs_dim)
        # hardcap is True
        if maxc >= 12 or not hardcap:
            return "STF given a,b"
        return "STF ab"



class StudentTFilter_Newton(basic.StudentTStateFilter):
    """
    An implementation of our proposed Student-t Filter using the newton method to find the fixed points 
    of :math:`a` and :math:`b`
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 nu : int = 1, current_time: float = None, seed: int = None, exp_hist_len: int = 100,
                 process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter

        :param nu: The degrees of freedom :math:`\nu` of the observation distribution. That is we 
                assume that in the system dynamics

                .. math:: 
                    x_t &= F_t(x_{t-1}) + v_t, &\quad v_t &\sim t_{\nu+n+m}\,(0, Q_t)\\
                    y_t &= H_t(x_t) + e_t,     &\quad e_t &\sim t_\nu\,(0, R_t).
                
                and our state hypothesis is also a student-t distribution :math:`x_{t} \sim t_{\nu + m}\,(\mu, \Sigma)`.
                Here :math:`n` is the dimension of :math:`x_t` and :math:`m` is the dimension of 
                :math:`y_t`. Note that :math:`\Sigma` is a scale matrix and not the covariance matrix.
        """
        super().__init__(model=model, mean=mean, covar=covar, state_nu=nu +model.obs_dim, current_time=current_time, 
                         seed=seed, exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["a"] = np.empty((exp_hist_len+1, 1))
        self._hist["a"][0] = 1
        self._hist["b"] = np.empty((exp_hist_len+1, 1))
        self._hist["b"][0] = 1
        self._process_scalar = utils.KLDmin_Norm_T(dim=self._state_dim, nu=self._state_nu+self._state_dim)
        self._obs_scalar = utils.KLDmin_Norm_T(dim=self._obs_dim, nu=nu)

        # The scale matrix this algorithm works on is self._state_scalar * self._state_covar
        # instead of the normal self._state_covar
        # _state_scalar is set by the parent StudentTStateFilter class

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:

        n, m, k = (self._state_dim, self._obs_dim, Fm.shape[0])
        Id_m = np.eye(m)
        nu = self._state_nu - m
        ss = self._state_scalar


        for i in range(k):
            res = obs[i] - HFm[i]

            #a = (nu + m) / (nu + (res.T @ np.linalg.solve(R[i], res))[0,0])
            a = (nu + m) / (nu + (res.T @ scipy.linalg.solve(R[i], res, check_finite=False, assume_a='pos'))[0,0])
            b = (nu + m + n) / (nu + m + max(n-2,0) *(nu+m)/(nu+m+2))
            # we have to counteract that P (from self._state_covar) is not yet the scale matrix
            
            for (a, b) in [(a, b)]: #[(a0_naive, b0_naive), (a0_naive, 0), (0, b0_naive), (0, 0), (a, b), ]:
                it = 0
                while it < 100:
                    #S_inv   = np.linalg.solve( (ss*a)*HFPFH[i] + (a*b)*HQH[i] + R[i], np.eye(m))
                    S_inv   = scipy.linalg.solve( (ss*a)*HFPFH[i] + (a*b)*HQH[i] + R[i], np.eye(m), check_finite=False, assume_a='pos')
                    Sr      = S_inv @ res
                    Rr      = R[i] @ Sr
                    HQHr    = HQH[i] @ Sr
                    HFPFHr  = ss* (HFPFH[i] @ Sr)
                    P2r     = HFPFHr + b* HQHr
                    SRr     = S_inv @ Rr
                    SHFPFHr = S_inv @ HFPFHr

                    RS      = R[i] @ S_inv
                    QS      = HQH[i] @ S_inv
                    PS      = HFPFH[i] @ S_inv
                    TraceRSPS = np.sum(RS * PS.T)*ss
                    TraceRSQS = np.sum(RS * QS.T)
                    TracePSPS = np.sum(PS * PS.T)*ss**2
                    TracePSQS = np.sum(PS * QS.T)*ss

                    # diffA S_inv = - S_inv @ (HFPFH + b*HQH) @ S_inv
                    # diffB S_inv = - S_inv @ (a*HQH) @ S_inv

                    cov_scale       = (nu + (res.T @ Sr)[0,0])/(nu + m -2)  #   res @ S_inv @ res
                    cov_scale_diffA = - (Sr.T @ P2r)[0,0]/(nu + m -2)       # - res @ S_inv @ (HFPFH + b*HQH) @ S_inv @ res
                    cov_scale_diffB = - a*(Sr.T @ HQHr)[0,0]/(nu + m -2)    # - res @ S_inv @ (a*HQH) @ S_inv @ res
                    traceA          = m - np.trace(RS)                      # m - Trace(R @ S_inv)
                    traceA_diffA    = TraceRSPS + b*TraceRSQS               # Trace(R @ S_inv @ (HFPFH+b*HQH) @ S_inv)
                    traceA_diffB    = a*TraceRSQS                           # Trace(R @ S_inv @ (a*HQH) @ S_inv)
                    traceB          = a*(n - a*np.trace(PS))                # a*n - a**2 *Trace( HFPFH @ S_inv )
                    traceB_diffA    = (n - 2*a*np.trace(PS)                 #   n - 2*a  *Trace( HFPFH @ S_inv )
                                    + a**2 * (TracePSPS + b*TracePSQS))     #     + a**2 *Trace( HFPFH @ S_inv @ (HFPFH + b*HQH) @ S_inv )
                    traceB_diffB    = a**3 * TracePSQS                      # a**2 *Trace( HFPFH @ S_inv @ (a*HQH) @ S_inv )
                    expectA         = (Sr.T @ Rr)[0,0]                      # res @ S_inv @ R @ S_inv @ res
                    expectA_diffA   = -2 * ( P2r.T @ SRr )[0,0]             # -  res @ S_inv @ (HFPFH + b*HQH) @ S_inv @ R @ S_inv @ res - res @ S_inv @ R @ S_inv @ (HFPFH + b*HQH) @ S_inv @ res
                    expectA_diffB   = -2 * a *( HQHr.T @ SRr )[0,0]         # -2*res @ S_inv @ (a*HQH) @ S_inv @ R @ S_inv @ res
                    rPr = (Sr.T @ HFPFHr)[0,0]
                    expectB         = a**2 * rPr                                # a**2 * res @ S_inv @ HFPFH @ S_inv @ res
                    expectB_diffA   = 2*a*rPr - 2*a**2 * (P2r.T @ SHFPFHr)[0,0] # 2a * res @ S_inv @ HFPFH @ S_inv @ res - a**2 * 2 * res @ S_inv @ (HFPFH + b*HQH) @ S_inv @ HFPFH @ S_inv @ res
                    expectB_diffB   = -2 * a**3 * (HQHr.T @ SHFPFHr)[0,0]       # -2*a**2 * res @ S_inv @ (a*HQH) @ S_inv @ HFPFH @ S_inv @ res

                    # We are looking at solutions (da, db) of the linearised Problem
                    #
                    #   | fa |     | fa_diffA    fa_diffB |   | da |     | 0 |
                    #   |    |  +  |                      | * |    |  =  |   |
                    #   | fb |     | fb_diffA    fb_diffB |   | db |     | 0 |
                    #
                    # Then the new values of a and b are a+da, b+db, where da and db can be found by inverting the 
                    # 2x2 Jacobi matrix, ie. dab = - Jf^-1 fab

                    fa_base         = nu + cov_scale*traceA/2 + expectA
                    fa              =  (nu + m)/fa_base - a
                    fa_diffA        = -(nu + m)/fa_base**2 * ( cov_scale_diffA*traceA/2 + cov_scale*traceA_diffA/2 + expectA_diffA ) -1
                    fa_diffB        = -(nu + m)/fa_base**2 * ( cov_scale_diffB*traceA/2 + cov_scale*traceA_diffB/2 + expectA_diffB )
                    fb_base         = nu + m + cov_scale*traceB/2 + expectB
                    fb              =  (nu + m + n)/fb_base - b
                    fb_diffA        = -(nu + m + n)/fb_base**2 * ( cov_scale_diffA*traceB/2 + cov_scale*traceB_diffA/2 + expectB_diffA )
                    fb_diffB        = -(nu + m + n)/fb_base**2 * ( cov_scale_diffB*traceB/2 + cov_scale*traceB_diffB/2 + expectB_diffB ) -1
                    
                    # Note that a 2x2 matrix [[A B], [C D]] has determinant det = AD-BC
                    det = fa_diffA*fb_diffB - fa_diffB*fb_diffA
                    if abs(det) < 1e-8:
                        print("WARNING! SINGULAR JACOBI")
                    # and [[A B], [C D]]^-1 = 1/det [[D -B], [-C A]]
                    # Thus the inverse Jacobi is [[ fb_diffB, -fa_diffB]
                    #                             [-fb_diffA,  fa_diffA]] /det
                    a_approx = a - ( fb_diffB*fa - fa_diffB*fb)/det
                    b_approx = b - (-fb_diffA*fa + fa_diffA*fb)/det
                    if a_approx < 0 or a_approx > (nu+m)/nu:
                        # If the newton method results in an a estimate outside of the domain of a
                        # we will instead use the fixed point method for the next a estimate
                        if b_approx < 0 or b_approx > (nu+m+n)/(nu+m):
                        #    print("Warning: out of bounds in Newton method for a and b")
                            b_approx = b+fb
                        #else:
                        #    print("Warning: out of bounds in Newton method for a")
                        a_approx = a+fa
                    elif b_approx < 0 or b_approx > (nu+m+n)/(nu+m):
                        #print("Warning: out of bounds in Newton method for b")
                        b_approx = b+fb
                    
                    if abs(a_approx - a)/(a_approx) + abs(b_approx - b)/(b_approx) < 1e-2:
                        break
                    a, b, it = (a_approx, b_approx, it+1)

            P_pri  = ((ss*a)*FPF[i]  + (a*b)*Q[i])
            P_priH = ((ss*a)*FPFH[i] + (a*b)*QH[i])
            K = P_priH @ S_inv

            self._state_mean[i] = Fm[i] + K @ res
            c = (nu + (res.T @ Sr)[0,0]) / (nu + m) / ss
            self._state_covar[i] = c * (P_pri - K @ P_priH.T)
            #self._state_covar[i] = (self._state_covar[i] + self._state_covar[i].T)/2
        return {"iterations": [it], "a":[np.squeeze(a)], "b":[np.squeeze(b)]}

    @staticmethod
    def label() -> str:
        return "STF NWT our"
    
    def label_long(self) -> str:
        return "our STF Newton"

    def desc(self) -> str:
        return "proposed Student-t Filter via Newtons method, nu = {:}".format(self._state_nu-self._obs_dim)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc >= 17:
            return r"STF NWT proposed, $\nu$={:}".format(self._state_nu-self._obs_dim)
        # hardcap is True
        if maxc >= 12 or not hardcap:
            return "STF NWT proposed"
        return "STF NWT"



class StudentTFilter_SF(basic.StudentTStateFilter):
    """
    An implementation of our proposed Student-t Filter adapted for sensor fusion. The difference is
    that this filter considers that outliers in compartments of the observation occur independent of
    each other - for example if the measurements of multiple sensors should be fused then it is a
    reasonable assumption that outliers in different sensors occur independently. 
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 nu : int = 1, comp: list = None, current_time: float = None, seed: int = None, 
                 exp_hist_len: int = 100, process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter.
        
        We assume the system dynamics where the observation noise :math:`e_t` has independently 
        distributed components :math:`e_t^1, ..., e_t^K`

        .. math:: 
            x_t &= F_t(x_{t-1}) + v_t, &\quad v_t &\sim t_{\nu+n+m}\,(0, Q_t)\\
            y_t &= H_t(x_t) + e_t,     &&\\
            e_t &= \left( \begin{array}{c} e_t^1 \\ \vdots \\ e_t^K \end{array} \right) &
                \quad e_t^i &\sim t_{\nu+\sum_{j=1}^{i-1} p_j}\,(0, R_t^i),\ i = 1,...,K
        
        and our state hypothesis is also a student-t distribution :math:`x_{t} \sim t_{\nu + m}\,(\mu, \Sigma)`.
        Here :math:`n` is the dimension of :math:`x_t` and :math:`m` is the dimension of 
        :math:`y_t` and the :math:`R_t^i`. Note that :math:`\Sigma` is a scale matrix and not the
        covariance matrix.

        :param nu: The degrees of freedom :math:`\nu` of the observation distribution.
        :param comp: A list of integers giving the lengths :math:`p_j` of the :math:`i`th independent 
                component of the observations.
                Thus ``sum(comp)`` should equal to :math:`m`. If ``None`` assume a single component, 
                i.e. there are no indepedendent subcomponents.
        """
        super().__init__(model=model, mean=mean, covar=covar, state_nu=nu +model.obs_dim, current_time=current_time, 
                         seed=seed, exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["a"] = np.empty((exp_hist_len+1, 1))
        self._hist["a"][0] = 1
        self._hist["b"] = np.empty((exp_hist_len+1, 1))
        self._hist["b"][0] = 1
        self._process_scalar = utils.KLDmin_Norm_T(dim=self._state_dim, nu=self._state_nu+self._state_dim)
        self._obs_scalar = utils.KLDmin_Norm_T(dim=self._obs_dim, nu=nu)
        comp = [self._obs_dim] if comp is None else comp
        assert sum(comp) == self._obs_dim, f"The components lengths {comp} should sum up to {self._obs_dim}!"
        start, idxs = (0, [0])
        for l in comp:
            start += l
            idxs.append(start)
        self._complens = comp
        self._compidxs = idxs
        self._eK = len(comp)

        # The scale matrix this algorithm works on is self._state_scalar * self._state_covar
        # instead of the normal self._state_covar
        # _state_scalar is set by the parent StudentTStateFilter class
    
    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:

        n, m, k = (self._state_dim, self._obs_dim, Fm.shape[0])
        Id_m = np.eye(m)
        nu = self._state_nu - m
        ss = self._state_scalar
        eK = self._eK

        compidxs = self._compidxs
        complens = self._complens

        for i in range(k):
            res = obs[i] - HFm[i]
            # The blocks on the diagonal of R
            Ris    = [R[i,compidxs[idx]:compidxs[idx+1],compidxs[idx]:compidxs[idx+1]] 
                      for idx in range(eK)]
            # The residuals with respect to each component (Here not adapted by S but during the iterations)
            Srs  = [res[compidxs[idx]:compidxs[idx+1]] 
                      for idx in range(eK)]
            # The first estimate on the Mahalanobis distance of the observation noises
            eiReis = [(Srs[idx].T @ scipy.linalg.solve(Ris[idx], Srs[idx], check_finite=False, assume_a='pos'))[0,0] 
                      for idx in range(eK)]
            # remember that we assumed that the e_i have distribution t_{nu+ p?}(0, Ris[i]), where
            # the additional dimension p? is the sum of the dimension of the prior components
            # This assumes that the observations in the first component are most likely to produce 
            # outliers and the observations in the last component are the least likely.
            # To make this more realistic we reorder the components into an order that we assume follows
            # how strongly pronounced the outlier in each component is (only for the current observation)
            order = np.argsort(eiReis)[::-1]
            phis  = np.ones((eK,))
            prior_dims = [0]
            #print(f"eiReis: {eiReis} order: {order}")
            for idx in range(eK-1):
                prior_dims.append(prior_dims[-1]+complens[order[idx]])
                # the scaling parameters phi[i+1] = r1 * ... * ri wuth
                # ri = (nu + sum_{j=1}^i p_j) / (nu + sum_{j=1}^{i-1} p_j + eiRei)
                # here i is order[idx]
                phis[order[idx+1]] = phis[order[idx]] * (nu + prior_dims[idx+1]) / (nu + prior_dims[idx] + eiReis[order[idx]])

            #print("init phis: ", phis)
            a_sum = 0
            for idx in range(eK):
                a_sum += eiReis[idx]/phis[idx]
            a = (nu + m) / (nu + a_sum)
            b = (nu + m + n) / (nu + m + max(n-2,0) *(nu+m)/(nu+m+2))
            
            it = 0
            while it < 100:
                S = (a*ss)*HFPFH[i] + (a*b)*HQH[i]
                for idx in range(eK):
                    S[compidxs[idx]:compidxs[idx+1], compidxs[idx]:compidxs[idx+1]] += phis[idx]*Ris[idx]
                #SdivA_inv = np.linalg.solve(SdivA, Id_m)
                S_inv = scipy.linalg.solve( S, Id_m, check_finite=False, assume_a='pos')
                Sr = S_inv @ res
                cov_scale  = (nu + (res.T @ Sr)[0,0])/(nu + m -2)

                # reuse the res_s to now hold the components of the updated residual S_inv @ res
                Srs    = [Sr[compidxs[idx]:compidxs[idx+1]] for idx in range(eK)]
                eiReis = [cov_scale/2*phis[idx]*(complens[idx] - phis[idx]*np.sum(Ris[idx]*
                                S_inv[compidxs[idx]:compidxs[idx+1],compidxs[idx]:compidxs[idx+1]]))
                          + phis[idx]**2 *(Srs[idx].T @ Ris[idx] @ Srs[idx])[0,0] 
                          for idx in range(eK)]

                #a_approx = (nu + m) / (nu + cov_scale*(m - np.sum(R_new * S_inv))/2 + (Sr.T @ R_new @ Sr)[0,0])
                a_sum = 0
                for idx in range(eK):
                    a_sum += eiReis[idx]/phis[idx]
                a_approx = (nu + m) / (nu + a_sum)
                b_approx = (nu + m + n) / (nu + m + cov_scale*a*(n  - (ss*a)*np.sum(HFPFH[i] * S_inv))/2 + (ss*a*a) * (Sr.T @ HFPFH[i] @ Sr)[0,0])
                # update phi values:
                phi_error = 0
                phis[order[0]] = 1
                for idx in range(eK-1):
                    new_phi = phis[order[idx]] * (nu + prior_dims[idx+1]) / (nu + prior_dims[idx] + eiReis[order[idx]])
                    phi_error += abs(new_phi-phis[order[idx+1]])/phis[order[idx+1]]
                    phis[order[idx+1]] = new_phi
                
                if abs(a_approx - a)/(a_approx) + abs(b_approx - b)/(b_approx) + phi_error/eK < 1e-2:
                    break
                a, b, it = (a_approx, b_approx, it+1)

            P_pri  = ((ss*a)*FPF[i]  + (b*a)*Q[i])
            P_priH = ((ss*a)*FPFH[i] + (b*a)*QH[i])
            K = P_priH @ S_inv

            self._state_mean[i] = Fm[i] + K @ res
            c = (nu + (res.T @ Sr)[0,0]) / (nu + m) / ss
            self._state_covar[i] = c * (P_pri - K @ P_priH.T)
            self._state_covar[i] = (self._state_covar[i] + self._state_covar[i].T)/2
        return {"iterations": [it], "a":[np.squeeze(a)], "b":[np.squeeze(b)]}
    

    @staticmethod
    def label() -> str:
        return "STF SF our"

    def desc(self) -> str:
        return "proposed Student-t Filter for Sensor Fusion, nu = {:}".format(self._state_nu-self._obs_dim)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc >= 20:
            return r"STF SF proposed, $\nu$={:}".format(self._state_nu-self._obs_dim)
        # hardcap is True
        if maxc >= 15 or not hardcap:
            return "STF SF proposed"
        return "STF SF"



class chang_RKF_SF(basic.NormalStateFilter):
    """
    An adaptation of a robust Kalman Filter (RKF) proposed in ‘Robust Kalman Filtering Based on
    Mahalanobis Distance as Outlier Judging Criterion’, Guobin Chang, 2014. https://doi.org/10.1007/s00190-013-0690-8.
    Here we implement the same idea for Sensor Fusion as in :class:`StudentTFilter_SF` for a fair
    comparison.
    """

    def __init__(self, model: abstract.AbstractDynamicModel, mean: np.ndarray = None, covar: np.ndarray = None, 
                 current_time: float = None, comp: list = None, alpha=0.01, seed: int = None, 
                 exp_hist_len: int = 100, process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        r"""
        Constructs the filter

        :param alpha: Given an observation :math:`y` and expected observation :math:`\bar y` with 
                Mahalanobis distance :math:`\gamma = (y- \bar y)^\top (H P H^\top + \kappa R)^{-1} (y - \bar y)`,
                then the observation will be treated as an outlier if the probability of observing a greater
                Mahalanobis distance is less than ``alpha`` and :math:`\kappa \geq 1` is scaled up until the 
                probability is (nearly) exactly ``alpha``. Defaults to 0.01.

                For :math:`k` components we find :math:`\kappa_1, ..., \kappa_k` such that the 
                probability of observing greater Mahalanobis distances
                :math:`(y_j- \bar y_j)^\top ( (H P H)^\top_{jj} + \kappa_j R_{jj})^{-1} (y_j - \bar y_j)`` 
                is each less than ``\alpha``. Where :math:`(H P H)_{jj}` and :math:`R_{jj}`
                are the block matrixes on the diagonals corresponding to the component :math:`j` 
                
        :param comp: A list of integers giving the lengths :math:`p_j` of the :math:`i`th independent 
                component of the observations.
                Thus ``sum(comp)`` should equal to :math:`m`. If ``None`` assume a single component, 
                i.e. there are no indepedendent subcomponents.
        """
        super().__init__(model=model, mean=mean, covar=covar, current_time=current_time, seed=seed, 
                         exp_hist_len=exp_hist_len, process_scalar=process_scalar, obs_scalar=obs_scalar)
        self._alpha = alpha
        comp = [self._obs_dim] if comp is None else comp
        assert sum(comp) == self._obs_dim, f"The components lengths {comp} should sum up to {self._obs_dim}!"
        start, idxs = (0, [0])
        for l in comp:
            start += l
            idxs.append(start)
        self._complens = comp
        self._compidxs = idxs
        self._eK = len(comp)

        self._thres = [scipy.stats.chi2(l).ppf(1-alpha) for l in comp]
        k = self._state_mean.shape[0]
        self._hist["iterations"] = np.empty((exp_hist_len+1, 1))
        self._hist["iterations"][0] = 0
        self._hist["kappas"] = np.empty((exp_hist_len+1, k, self._eK))
        self._hist["kappas"][0] = np.ones((k, self._eK), dtype=float)

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        HPH = HFPFH + HQH
        it, k, m = (0, Fm.shape[0], self._obs_dim)
        eK = self._eK
        compidxs = self._compidxs

        kappa = np.ones((k,eK), dtype=float)
        for i in range(k):
            res = obs[i] - HFm[i]
            HPHi = HPH[i]
            Ris = [R[i,compidxs[idx]:compidxs[idx+1],compidxs[idx]:compidxs[idx+1]] 
                      for idx in range(eK)]
            smth_changed = True
            while smth_changed:
                smth_changed = False
                S = np.copy(HPHi)
                for idx in range(eK):
                    S[compidxs[idx]:compidxs[idx+1],compidxs[idx]:compidxs[idx+1]] += kappa[i,idx]*Ris[idx]
                Pn = scipy.linalg.solve(S, res, check_finite=False, assume_a='pos')
                gamma = [(res[compidxs[idx]:compidxs[idx+1]].T @ Pn[compidxs[idx]:compidxs[idx+1]])[0,0]
                        for idx in range(eK)]    # The Mahalanobis distances
                for idx in range(eK):
                    if gamma[idx] > self._thres[idx]:
                        smth_changed = True
                        kappa[i,idx] += (gamma[idx] - self._thres[idx]) /  (Pn[compidxs[idx]:compidxs[idx+1]].T @ Ris[idx] @ Pn[compidxs[idx]:compidxs[idx+1]])[0,0]
                it += 1
        
            PH  = FPFH[i] + QH[i]
            S = np.copy(HPHi)
            for idx in range(eK):
                S[compidxs[idx]:compidxs[idx+1],compidxs[idx]:compidxs[idx+1]] += kappa[i,idx]*Ris[idx]
            #PHS = PH @ np.linalg.solve(HPHi + kappa[i]*Ri, np.eye(m))
            PHS = PH @ scipy.linalg.solve(S, np.eye(m), check_finite=False, assume_a='pos')
            self._state_mean[i]  = Fm[i] + PHS @ res
            self._state_covar[i] = FPF[i]+Q[i] - PHS @ PH.T
            self._state_covar[i] = (self._state_covar[i] + self._state_covar[i].T)/2

        return {"iterations": [it], "kappas":[np.squeeze(kappa)]}

    @staticmethod
    def label() -> str:
        return "RKF SF"
    
    def label_long(self) -> str:
        return "RKF SF Chang"

    def desc(self) -> str:
        return "Robust Kalman Filter after Guobin Chang, 2014. Adapted for Sensorfusion. alpha = {:.2f}%".format(self._alpha*100)
    
    def desc_short(self, maxc: int = 20, hardcap: bool = False) -> str:
        if maxc > 18 or not hardcap:
            return r"RKF SF Chang, $\alpha$={:.1f}".format(self._alpha)
        # hardcap is True
        return "RKF SF Chang"[:hardcap]
