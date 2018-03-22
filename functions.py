import os
import numpy as np
from constance import GG


def get_rho(temp, sal):
    """
    Get density, Foffonoff state sea water
    :param temp:
    :param sal:
    :return: rho
    """
    print("get rho")
    R00 = 1000
    A = 999.842594 + 6.793952e-2*temp - 9.09529e-3*temp**2 + 1.001685e-4*temp**3 - 1.120083e-6*temp**4 + 6.536332e-9*temp**5
    B = 8.24493e-1 - 4.0899e-3*temp + 7.6438e-5*temp**2 - 8.2467e-7*temp**3 + 5.3875e-9*temp**4
    C = -5.72466e-3  +  1.0227e-4*temp - 1.6546e-6*temp**2
    D = 4.8314e-4
    E = 19652.21 + 148.4206*temp - 2.327105*temp**2 + 1.360477e-2*temp**3 - 5.155288e-5*temp**4
    F = 54.6746-.603459*temp  + 1.09987e-2*temp**2  -  6.167e-5*temp**3
    G = 7.944e-2+1.6483e-2*temp - 5.3009e-4*temp**2
    H = 3.239908  +  1.43713e-3*temp  +  1.16092e-4*temp**2 - 5.77905e-7*temp**3
    I = 2.2838e-3 - 1.0981e-5*temp - 1.6078e-6*temp**2
    J = 1.91075e-4
    M = 8.50935e-5 - 6.12293e-6*temp + 5.2787e-8*temp**2
    N = -9.9348e-7 + 2.0816e-8*temp +  9.1697e-10*temp**2
    R0 = A + B*sal + C*sal**1.5 + D*sal**2
    P = 0
    RK = E+F*sal+G*sal**1.5+(H+I*sal+J*sal**1.5)*P+(M+N*sal)*P**2
    rho = R0/(1-P/RK)-R00
    return rho


def get_bvf(rho, depth):
    """
    Get Brenta-Vaisala frequency
    :param rho:
    :param depth:
    :return: frequency
    """
    length = rho.shape[0]

    # init bvf array
    bvf = np.empty(length)
    bvf.fill(0)

    # incompressible fluid case
    r1 = rho[0]
    for i in range(length-1):
        r2 = rho[i+1]
        bvf[i] = np.sqrt(
            np.abs(
                GG*(r2 - r1)/(depth[i+1] - depth[i])/(1000 + r2)
            )
        )
        r1 = r2

    bvf[i+1] = bvf[i]

    return bvf
