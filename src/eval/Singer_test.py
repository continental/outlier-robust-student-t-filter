# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Tests the implementations of the :meth:`~src.filters.models.ExtendedSingerModel`, now less important since
the implementation shifted to using sympys evaluation for numerical stability reasons
"""

import numpy as np
from .. import filters, utils
import sympy as sym

print("-   Loading File 'Singer_test.py'")

if __name__ == "__main__":
    H = sym.Matrix([[0,], [0,], [1,]])

    a = sym.Symbol("\\alpha")
    b = sym.Symbol("\\beta")
    s = sym.Symbol("s")
    sig2 = sym.Symbol("\\sigma^2")
    t_diff = sym.Symbol(r"\Delta t\ ")

    for k in range(10): 
        # if the test should be repeated for multiple alpha/beta values
        # The first 3 test test for small alpha beta values
        while True:
            alpha  = 1e-8 if k in [0,2] else np.abs(np.random.normal())
            beta   = 1e-7 if k in [0,1] else np.abs(np.random.normal())
            sigma2 = 1.0 if k in [0,1,2] else np.random.normal()**2
            dt     = 1.1 if k in [0,1,2] else np.random.normal()**2
            if np.abs(alpha-beta)/alpha > 0.5:
                break
        print(f"Test ExtendedSingerModel for \n alpha   = {alpha:.8f},\n beta    = {beta:.8f},"+
            f"\n sigma^2 = {sigma2:.8f},\n dt      = {dt:.8f}: ")

        for i in range(3):
            ralpha, rbeta = [(alpha, beta), (alpha, alpha), (alpha, 0)][i]
            print([" - normal:", " - beta = alpha:", " - beta = 0:"][i] )
            SingerModel = filters.models.ExtendedSingerModel(alpha=ralpha, beta=rbeta, sigma2=sigma2, d=1)
            if i == 0:
                symF = sym.Matrix([[1, (1-sym.exp(-b*s))/b, (1-sym.exp(-a*s))/(b-a)/a - (1-sym.exp(-b*s))/(b-a)/b ],
                                [0, sym.exp(-b*s),       1/(b-a)*( (sym.exp(-a*s) - sym.exp(-b*s)) )           ],
                                [0, 0,                   sym.exp(-a*s)                                         ]])
            elif i == 1:
                symF = sym.Matrix([[1, (1-sym.exp(-a*s))/a, (1-sym.exp(-a*s)*(a*s+1))/a**2 ],
                                [0, sym.exp(-a*s),       s* (sym.exp(-a*s) )            ],
                                [0, 0,                   sym.exp(-a*s)                  ]])
            else:
                symF = sym.Matrix([[1, s, s/a - (1- sym.exp(-a*s))/a**2 ], 
                                [0, 1,       (1- sym.exp(-a*s))/a    ], 
                                [0, 0,           sym.exp(-a*s)       ]])
            evalF  = np.array(symF.evalf(30, subs={a: ralpha, b: rbeta, s: dt}), dtype=float)
            singF  = SingerModel._F(dt)
            errorF = np.abs(np.sum(singF - evalF))
            print(f" -  error in the transition model:               {errorF:.20e}")
            if errorF > 0.01*dt:
                print("    is:\n{}\n    but should be:\n{}".format(
                    utils.nd_to_str(singF, shift=5), 
                    utils.nd_to_str(evalF, shift=5)
                    ))

            if i == 0:
                hand00 = ( (
                            -(b**2 *(sym.exp(-2*t_diff*a)-1))/2/a
                            +(a-b)*( 2*a*(sym.exp(-t_diff*b)-1) )/b
                            -(a-b)*( 2*b*(sym.exp(-t_diff*a)-1) )/a
                            -(a**2 *(sym.exp(-2*t_diff*b)-1))/2/b
                            +(2*a*b*(sym.exp(-(a+b)*t_diff)-1))/(a+b)
                        )
                            + (a-b)**2 * t_diff
                        ) / (a**2 * b**2 * (a-b)**2)
                hand01 = ( (a-b) + b *sym.exp(-t_diff*a) - a *sym.exp(-t_diff*b) )**2/2/(a**2 *b**2 *(a-b)**2)
                hand02 = (2*sym.exp(-t_diff*(a+b)) *a**2 + (a-b)*b -2* sym.exp(-t_diff*a)*(a-b)*(a+b) - sym.exp(-2*t_diff*a)*b*(a+b))/(2*a**2 *(a-b) *b *(a+b))
                hand02 = ( a/(a+b)*(sym.exp(-t_diff*(a+b))-1) + (b/a-1)*(sym.exp(-t_diff*a)-1) - b/2/a*(sym.exp(-2*t_diff*a)-1) ) / (a*b*(a-b))
                hand11 = ((1-sym.exp(-2*b*t_diff))/2/b - 2*(1-sym.exp(-(a+b)*t_diff))/(a+b) + (1-sym.exp(-2*a*t_diff))/2/a) / (a-b)**2
                hand12 = ( (sym.exp(-(a+b)*t_diff)-1)/(a+b) - (sym.exp(-2*a*t_diff)-1)/(2*a) )/(b-a)
            elif i == 1:
                hand00 = (4*t_diff*a - 11 + 8*(t_diff*a + 2)*sym.exp(-t_diff*a) - (2*t_diff**2 *a**2 + 6*t_diff*a+5)*sym.exp(-2*t_diff*a))/(4*a**5)
                hand01 = (1- (t_diff*a+1)*sym.exp(-t_diff*a))**2 / (2*a**4)
                hand02 = ( (1-2*sym.exp(-t_diff*a))**2 - (1-2*t_diff*a)*sym.exp(-2*t_diff*a) )/(4*a**3)
                hand11 = ( 1- (2*t_diff**2*a**2 + 2*t_diff*a + 1)*sym.exp(-2*t_diff*a) ) / (4*a**3)
                hand12 = ( 1 - (2*t_diff*a+ 1)* sym.exp(-2*t_diff*a) ) / (4*a**2)
            elif i == 2:
                hand00 = ( 5 - 2*(1 - t_diff*a)**3 - 12*t_diff*a*sym.exp(-t_diff*a) - 3*sym.exp(-2*t_diff*a) )/(6*a**5)
                hand01 = ( (1 - t_diff*a - sym.exp(-t_diff*a))**2 )/(2*a**4)
                hand02 = ( (1 - 2*t_diff*a*sym.exp(-t_diff*a) - sym.exp(-2*t_diff*a)) )/(2*a**3)
                hand11 = ( (2*t_diff*a - 3 + 4*sym.exp(-t_diff*a) - sym.exp(-2*t_diff*a)) )/(2*a**3)
                hand12 = ( (1 - sym.exp(-t_diff*a))**2 )/(2*a**2)
            hand22 = 1/2/a - sym.exp(-2*t_diff*a)/2/a
            symQ = sym.Matrix([[hand00, hand01, hand02],
                                [hand01, hand11, hand12],
                                [hand02, hand12, hand22]])
            evalQ  = sigma2*np.array(symQ.evalf(30, subs={a: ralpha, b: rbeta, t_diff: dt}), dtype=float)
            singQ  = SingerModel._Q(dt)
            errorQ = np.sum(np.abs(singQ - evalQ)) / np.sum(np.abs(evalQ))
            print(f" -  error in the process noise covariance model: {errorQ:.20e}")
            singQeig = np.linalg.eig(singQ)[0]
            if errorQ > 0.1 or np.any(singQeig <= 0):
                evalQeig = np.linalg.eig(evalQ)[0]
                singQeig = np.linalg.eig(singQ)[0]
                print("    eigenvalues of symbolic solution are [{:.3e},{:.3e},{:.3e}]\n".format(
                        evalQeig[0], evalQeig[1], evalQeig[2]) +
                    "     but of implemented are [{:.3e},{:.3e},{:.3e}]".format(
                        singQeig[0], singQeig[1], singQeig[2]))
                if True:#np.all(evalQeig > 0) or np.any(singQeig < 0):
                    print("    is:\n{}\n    but should be:\n{}".format(
                        utils.nd_to_str(singQ, shift=5),
                        utils.nd_to_str(evalQ, shift=5),
                        ))