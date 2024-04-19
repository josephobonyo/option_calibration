import math
import numpy as np
import pandas as pd
import scipy.optimize as sop
import matplotlib.pyplot as plt
import matplotlib as mpl
from bsm_functions_GK import bsm_call_value_GK

def M76_option_analytic(type, S0, K, T, r, y, sigma, lamb, theta, delta, nterms):
    ''' Valuation of European call option in M76 model via
    analytic formula derived in Mazzoni

    Parameters
    ==========
    type: int
        call if 0, put if 1
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    y: float
        constant dividend rate
    sigma: float
        volatility factor in diffusion term
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    nterms: int
        number of terms in the sum

    Returns
    =======
    call_value: float
        European call option present value
    '''

    k = np.exp(theta+0.5*delta**2)-1
    mu = r-y-k*lamb
    call = 0.0
    lamb_tilde = lamb + lamb * k
    explamb = np.exp(-lamb_tilde*T)
    for n in range(nterms):
        r_n = r + n*(theta+0.5*delta**2)/T - k*lamb
        sigma_n = np.sqrt(sigma**2+n*delta**2/T)
        bsm = bsm_call_value_GK(S0,K,T,r_n,y,sigma_n)
        call += bsm*explamb*(lamb_tilde*T)**n/math.factorial(n)
    if type == 0:
        value = call
    else:
        value = call - (S0*np.exp(-y*T)-K*np.exp(-r*T))
    return value
