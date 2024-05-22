# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Tests the distribution classes to make sure that they show expected behaviour
"""

import numpy as np
from .. import filters
import scipy

print("-   Loading File 'Distr_test.py'")

if __name__ == "__main__":

    Nbr_tests = 100000
    dims = [1, 3, 14]
    seed  = np.random.randint(2147483647)
    rng = np.random.default_rng(seed)
    rng_scipy = np.random.default_rng(rng.integers(2147483647))
    rng_impl  = np.random.default_rng(rng.integers(2147483647))
    print(f"\n\nseed: {seed}\n")

    print("testing pdf/logpdf of the normal distribution")
    for dim in dims:
        print(f"dimension {dim}: ")
        b = rng.normal(size=(dim,))
        E = rng.normal(size=(dim,dim))
        E = E @ E.T/dim
        b2 = rng.normal(size=(dim,))
        E2 = rng.normal(size=(dim,dim))
        E2 = E2 @ E2.T/dim

        distr_scipy    = scipy.stats.multivariate_normal(mean=b,  cov=E,  seed=rng_scipy.integers(2147483647))
        distr_scipy2   = scipy.stats.multivariate_normal(mean=b2, cov=E2, seed=rng_scipy.integers(2147483647))
        distr_impl     = filters.NormalDistribution(mu=b,  P=E,  seed=rng_impl)
        distr_impl.stack(filters.NormalDistribution(mu=b2, P=E2, seed=rng_impl) )


        eval_point  = distr_scipy.rvs( size=Nbr_tests).reshape((Nbr_tests,dim))
        eval_point2 = distr_scipy2.rvs(size=Nbr_tests).reshape((Nbr_tests,dim))
        eval_point_conc = np.stack((eval_point.T, eval_point2.T), axis=0)
        eval_point_impl = distr_impl.rvs(m=Nbr_tests)
        pdf_scipy   = np.stack((distr_scipy.pdf(eval_point), distr_scipy2.pdf(eval_point2)), axis=0)
        pdf_impl    = distr_impl.pdf(eval_point_conc)
        lpdf_scipy  = np.stack((distr_scipy.logpdf(eval_point), distr_scipy2.logpdf(eval_point2)), axis=0)
        lpdf_impl   = distr_impl.logpdf(eval_point_conc)
        #print(" - pdf_scipy: ", pdf_scipy)
        #print(" - pdf_impl:  ", pdf_impl)
        print(f" - max relative pdf error:    ["+" ".join([f"{perc:.4e}" for perc in np.max(np.abs(pdf_scipy-pdf_impl)/np.minimum(pdf_impl, pdf_scipy), axis=1)])+"]")
        #print(" - logpdf_scipy: ", lpdf_scipy)
        #print(" - logpdf_impl:  ", lpdf_impl)
        print(f" - max relative logpdf error: ["+" ".join([f"{perc:.4e}" for perc in np.max(np.abs(lpdf_scipy-lpdf_impl)/np.minimum(-lpdf_impl, -lpdf_scipy), axis=1)])+"]")
        # empirical covariance
        Q_scipy1 = np.atleast_2d(eval_point)-b[None,:]
        Q_scipy1 = np.sum(Q_scipy1[:, :,None]*Q_scipy1[:, None, :], axis=0)/Nbr_tests
        Q_scipy2 = np.atleast_2d(eval_point2)-b2[None,:]
        Q_scipy2 = np.sum(Q_scipy2[:, :,None]*Q_scipy2[:, None, :], axis=0)/Nbr_tests
        print(f" - max relative scipy error in estimated covariance with {Nbr_tests:} samples: [{np.max(np.abs(Q_scipy1-E))/np.max(E)*100:.2f}% {np.max(np.abs(Q_scipy2-E2))/np.max(E2)*100:.2f}%]")
        Ec, bc = (np.stack((E,E2)), np.stack((b,b2)))
        Q_impl = np.atleast_2d(eval_point_impl-bc[:,:,None])
        Q_impl = np.sum(Q_impl[:, None, :, :]*Q_impl[:, :, None, :], axis=3)/Nbr_tests
        print(f" - max relative impl  error in estimated covariance with {Nbr_tests:} samples: ["+"% ".join([f"{perc:.2f}" for perc in np.max(np.abs(Q_impl-Ec), axis=(1,2))/np.max(Ec)*100])+"%]")


    print("\ntesting pdf/logpdf of the student-t distribution")
    for nu in [20]:
        print("nu = ", nu)
        for dim in dims:
            print(f"dimension {dim}: ")
            b = rng.normal(size=(dim,))
            E = rng.normal(size=(dim,dim))
            E = E @ E.T/dim
            b2 = rng.normal(size=(dim,))
            E2 = rng.normal(size=(dim,dim))
            E2 = E2 @ E2.T/dim
            
            distr_scipy    = scipy.stats.multivariate_t(loc=b,  shape=E,  df=nu, seed=rng_scipy.integers(2147483647))
            distr_scipy2   = scipy.stats.multivariate_t(loc=b2, shape=E2, df=nu, seed=rng_scipy.integers(2147483647))
            distr_impl     = filters.StudentTDistribution(mu=b,  P=E,  nu=nu, seed=rng_impl)
            distr_impl.stack(filters.StudentTDistribution(mu=b2, P=E2, nu=nu, seed=rng_impl) )

            eval_point  = distr_scipy.rvs( size=Nbr_tests).reshape((Nbr_tests,dim))
            eval_point2 = distr_scipy2.rvs(size=Nbr_tests).reshape((Nbr_tests,dim))
            eval_point_conc = np.stack((eval_point.T, eval_point2.T), axis=0)
            eval_point_impl = distr_impl.rvs(m=Nbr_tests)
            pdf_scipy   = np.stack((distr_scipy.pdf(eval_point), distr_scipy2.pdf(eval_point2)), axis=0)
            pdf_impl    = distr_impl.pdf(eval_point_conc)
            lpdf_scipy  = np.stack((distr_scipy.logpdf(eval_point), distr_scipy2.logpdf(eval_point2)), axis=0)
            lpdf_impl   = distr_impl.logpdf(eval_point_conc)
            #print(" - pdf_scipy: ", pdf_scipy)
            #print(" - pdf_impl:  ", pdf_impl)
            print(f" - max relative pdf error:    ["+" ".join([f"{perc:.4e}" for perc in np.max(np.abs(pdf_scipy-pdf_impl)/np.minimum(pdf_impl, pdf_scipy), axis=1)])+"]")
            #print(" - logpdf_scipy: ", lpdf_scipy)
            #print(" - logpdf_impl:  ", lpdf_impl)
            print(f" - max relative logpdf error: ["+" ".join([f"{perc:.4e}" for perc in np.max(np.abs(lpdf_scipy-lpdf_impl)/np.minimum(-lpdf_impl, -lpdf_scipy), axis=1)])+"]")
            # empirical covariance
            Q_scipy1 = np.atleast_2d(eval_point)-b[None,:]
            Q_scipy1 = np.sum(Q_scipy1[:, :,None]*Q_scipy1[:, None, :], axis=0)/Nbr_tests
            Q_scipy2 = np.atleast_2d(eval_point2)-b2[None,:]
            Q_scipy2 = np.sum(Q_scipy2[:, :,None]*Q_scipy2[:, None, :], axis=0)/Nbr_tests
            print(f" - max relative scipy error in estimated covariance with {Nbr_tests:} samples: [{np.max(np.abs(Q_scipy1-nu/(nu-2)*E))/np.max(nu/(nu-2)*E)*100:.2f}% {np.max(np.abs(Q_scipy2-nu/(nu-2)*E2))/np.max(nu/(nu-2)*E2)*100:.2f}%]")
            Ec, bc = (np.stack((E,E2)), np.stack((b,b2)))
            Q_impl = np.atleast_2d(eval_point_impl-bc[:,:,None])
            Q_impl = np.sum(Q_impl[:, None, :, :]*Q_impl[:, :, None, :], axis=3)/Nbr_tests
            print(f" - max relative impl  error in estimated covariance with {Nbr_tests:} samples: ["+"% ".join([f"{perc:.2f}" for perc in np.max(np.abs(Q_impl-Ec*nu/(nu-2)), axis=(1,2))/np.max(nu/(nu-2)*Ec)*100])+"%]")