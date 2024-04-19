import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import numpy as np
import scipy.stats as st
import random
import time
import datetime as dtp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels as stats
from bsm_functions_GK import bsm_call_value_GK
import timeit
from scipy.integrate import quad

def generate_cholesky(rho):
    ''' Function to generate Cholesky matrix.

    Parameters
    ==========
    rho: float
        correlation between index level and variance

    Returns
    =======
    matrix: NumPy array
        Cholesky matrix
    '''
    rho_rs = 0  # correlation between index level and short rate
    covariance = np.zeros((4, 4), dtype=float)
    covariance[0] = [1.0, rho_rs, 0.0, 0.0]
    covariance[1] = [rho_rs, 1.0, rho, 0.0]
    covariance[2] = [0.0, rho, 1.0, 0.0]
    covariance[3] = [0.0, 0.0, 0.0, 1.0]
    cho_matrix = np.linalg.cholesky(covariance)
    return cho_matrix


def random_number_generator(M, I, anti_paths, moment_matching):
    ''' Function to generate pseudo-random numbers.

    Parameters
    ==========
    M: int
        time steps
    I: int
        number of simulation paths
    anti_paths: bool
        flag for antithetic paths
    moment_matching: bool
        flag for moment matching

    Returns
    =======
    rand: NumPy array
        random number array
    '''
    if anti_paths:
        rand = np.random.standard_normal((4, M + 1, int(I / 2)))
        rand = np.concatenate((rand, -rand), 2)
    else:
        rand = np.random.standard_normal((4, M + 1, I))
    if moment_matching:
        for a in range(4):
            rand[a] = rand[a] / np.std(rand[a])
            rand[a] = rand[a] - np.mean(rand[a])
    return rand

#
# Function for Short Rate and Volatility Processes
#


def SRD_generate_paths(x0, kappa, theta, sigma, T, M, I,
                       rand, row, cho_matrix):
    ''' Function to simulate Square-Root Diffusion (SRD/CIR) process.

    Parameters
    ==========
    x0: float
        initial value
    kappa: float
        mean-reversion factor
    theta: float
        long-run mean
    sigma: float
        volatility factor
    T: float
        final date/time horizon
    M: int
        number of time steps
    I: int
        number of paths
    row: int
        row number for random numbers
    cho_matrix: NumPy array
        cholesky matrix

    Returns
    =======
    x: NumPy array
        simulated variance paths
    '''
    dt = T / M
    x = np.zeros((M + 1, I), dtype=float)
    x[0] = x0
    xh = np.zeros_like(x)
    xh[0] = x0
    sdt = np.sqrt(dt)
    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        xh[t] = (xh[t - 1] + kappa * (theta -
                                      np.maximum(0, xh[t - 1])) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[row] * sdt)
        x[t] = np.maximum(0, xh[t])
    return x

#
# Function for B96 Index Process
#


def B96_generate_paths(S0, r, y, v, lamb, mu, delta, rand, row1, row2,
                       cho_matrix, T, M, I, moment_matching):
    ''' Simulation of Bates (1996) index process.

    Parameters
    ==========
    S0: float
        initial value
    r: NumPy array
        simulated short rate paths
    y: float
        constant dividend yield
    v: NumPy array
        simulated variance paths
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    rand: NumPy array
        random number array
    row1, row2: int
        rows/matrices of random number array to use
    cho_matrix: NumPy array
        Cholesky matrix
    T: float
        time horizon, maturity
    M: int
        number of time intervals, steps
    I: int
        number of paths to simulate
    moment_matching: bool
        flag for moment matching

    Returns
    =======
    S: NumPy array
        simulated index level paths
    '''
    S = np.zeros((M + 1, I), dtype=float)
    S[0] = S0
    dt = T / M
    sdt = np.sqrt(dt)
    ranp = np.random.poisson(lamb * dt, (M + 1, I))
    bias = 0.0
    for t in range(1, M + 1, 1):
        ran = np.dot(cho_matrix, rand[:, t, :])
        if moment_matching:
            bias = np.mean(np.sqrt(v[t]) * ran[row1] * sdt)
        S[t] = S[t - 1] * (np.exp(((r[t] + r[t - 1]) / 2 - y - 0.5 * v[t]) * dt +
                                  np.sqrt(v[t]) * ran[row1] * sdt - bias) +
                           (np.exp(mu + delta * ran[row2]) - 1) * ranp[t])
    return S

def BCC97_lsm_valuation(type, S, r, v, K, T, M, I):
    ''' Function to value American options by LSM algorithm.

    Parameters
    ==========
    type: int
        0 if call, 1 if put
    S: NumPy array
        simulated index level paths
    r: NumPy array
        simulated short rate paths
    v: NumPy array
        simulated variance paths
    K: float
        strike of the put option
    T: float
        final date/time horizon
    M: int
        number of time steps
    I: int
        number of paths

    Returns
    =======
    LSM_value: float
        LSM Monte Carlo estimator of American option value
    '''
    dt = T / M
    D = 10
    # inner value matrix
    if type == 1:
        h = np.maximum(K - S, 0)
        # value/cash flow matrix
        V = np.maximum(K - S, 0)
    else:
        h = np.maximum(S - K, 0)
        # value/cash flow matrix
        V = np.maximum(S - K, 0)

    for t in range(M - 1, 0, -1):
        df = np.exp(-(r[t] + r[t + 1]) / 2 * dt)
        # select only ITM paths
        itm = np.greater(h[t], 0)
        relevant = np.nonzero(itm)
        rel_S = np.compress(itm, S[t])
        no_itm = len(rel_S)
        if no_itm == 0:
            cv = np.zeros((I), dtype=float)
        else:
            rel_v = np.compress(itm, v[t])
            rel_r = np.compress(itm, r[t])
            rel_V = (np.compress(itm, V[t + 1]) *
                     np.compress(itm, df))
            matrix = np.zeros((D + 1, no_itm), dtype=float)
            matrix[10] = rel_S * rel_v * rel_r
            matrix[9] = rel_S * rel_v
            matrix[8] = rel_S * rel_r
            matrix[7] = rel_v * rel_r
            matrix[6] = rel_S ** 2
            matrix[5] = rel_v ** 2
            matrix[4] = rel_r ** 2
            matrix[3] = rel_S
            matrix[2] = rel_v
            matrix[1] = rel_r
            matrix[0] = 1
            reg = np.linalg.lstsq(matrix.transpose(), rel_V)
            cv = np.dot(reg[0], matrix)
        erg = np.zeros((I), dtype=float)
        np.put(erg, relevant, cv)
        V[t] = np.where(h[t] > erg, h[t], V[t + 1] * df)
        # exercise decision
    df = np.exp(-((r[0] + r[1]) / 2) * dt)
    LSM_value = max(np.sum(V[1, :] * df) / I, h[0, 0])   # LSM estimator
    return LSM_value

np.random.seed(100000)
T = 1.0
M = 50
I = 10000
r0 = 0.04
S0 = 100
K = 101
type = 0
v0 = 0.2
y = 0.03 # dividend yield
kappa_r = 1.0
theta_r = 0.042
sigma_r = 0.1
kappa_v = 2.0
theta_v = 0.04
sigma_v = 0.2
lamb = 0.25
mu = -0.1
delta = 0.15
moment_matching = True
anti_paths = True
rho = -0.7 # correlation between variance process and index process
rand = random_number_generator(M, I, anti_paths, moment_matching)
cho_matrix = generate_cholesky(rho)
r = SRD_generate_paths(r0, kappa_r, theta_r, sigma_r, T, M, I,
                       rand, 0, cho_matrix)
v = SRD_generate_paths(v0, kappa_v, theta_v, sigma_v, T, M, I,
                       rand, 2, cho_matrix)
S = B96_generate_paths(S0, r, y, v, lamb, mu, delta, rand, 1, 3,
                       cho_matrix, T, M, I, moment_matching)

LSM_value = BCC97_lsm_valuation(type, S, r, v, K, T, M, I)
print("LSM value =    ", "%10.5f" % LSM_value )
