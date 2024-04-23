# Advanced Methods Calibration Project
# Joseph Obonyo

import numpy as np
import pandas as pd
import scipy.optimize as sop
import matplotlib.pyplot as plt
from BCC_option_valuation import BCC_call_value, H93_call_value
from BCCJ_option_valuation import BCCJ_call_value
from CIR_zcb_valuation_gen import B
from M76_analytic import M76_option_analytic
from MCExamples import (generate_cholesky, random_number_generator, 
                SRD_generate_paths, B96_generate_paths, BCC97_lsm_valuation)   # Functions for Monte Carlo Simulation
import os 

file_path = r""   # Enter file path to excel file
save_folder = r""   # Enter path to desired save location for graphs


S0 = 5123.41
df = pd.read_excel(file_path, sheet_name='calibration_sheet')
exptimes = np.array(df["exptimes"])
strikes = np.array(df["strike"])
calls = np.array(df["call_price"])
puts = np.array(df["put_price"])
rates = np.array(df["termrates"])
D_i = np.exp(-rates * exptimes)
v1 = np.array(df['call_iv'])
labels = list(df["label"])   # Label for time to expiration (3m, 6m, etc...)


def opt_y(f, Df, x0, epsilon, max_iter):
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        
        new = xn - fxn/Dfxn
        #print(new)
        if abs(new - xn) < epsilon:
            print("Found solution after", n, "iterations.")
            return new
        xn = new
        
    print('Exceeded maximum iterations. No solution found.')
    return None


# Bates Model Calibration
def BCC_error_function(p0, strikes, exptimes, calls, puts, rates, y):
    ''' Error Function for parameter calibration in BCC Model 

    Parameters as part of p0
    ==========
    sigma: float
        volatility factor in diffusion term
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    strikes: array of floats
        strike prices of the options
    T: float
        time to expiration
    calls: array of floats
        market call prices
    r: float
        risk-free interest rate
    y: float
        continuous dividend yield

    Returns
    =======
    RMSE: float
        root mean squared error
    '''
    global i, min_RMSE
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta  = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or \
            rho < -1.0 or rho > 1.0 or delta < 0.0 or lamb < 0.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 500.0
    L = len(strikes)
    se = []
    #S = S0*np.exp(-y*T)
    for j in range(L):
        T = exptimes[j]
        S = S0*np.exp(-y*T)
        K = strikes[j]
        r = rates[j]
        call_model_value = BCC_call_value(S, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
        put_model_value = call_model_value - S + K * np.exp(-r * T)
        se.append((call_model_value - calls[j])**2 + (put_model_value - puts[j])**2)
    RMSE = np.sqrt(sum(se))
    #RMSE = np.sqrt(sum(se) / len(se))
    min_RMSE = min(min_RMSE, RMSE)
    if i % 50 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    i += 1
    return RMSE


def generate_plot_BCC(opt,strikes,calls,puts,T,r,y,label):
    #
    # Calculating Model Prices
    #
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta  = opt
    
    call_model_values = []
    put_model_values = []
    call_diffs = []
    put_diffs = []
    L = len(strikes)
    S = S0*np.exp(-y*T)
    for j in range(L):
        K = strikes[j]
        call_model_values.append(BCC_call_value(S, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta))
        call_diffs.append(calls[j]-call_model_values[j])
        put_model_values.append(call_model_values[j] - S + K * np.exp(-r * T))
        put_diffs.append(puts[j]-put_model_values[j])
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, call_model_values, label='call model')
    plt2 = plt.plot(strikes, calls, label='call market')
    plt3 = plt.plot(strikes, put_model_values, label='put model')
    plt4 = plt.plot(strikes, puts, label='put market')
    plt.title('Bates: Market Prices Versus Calibrated Prices ' + label)
    plt.xlabel('strike')
    plt.ylabel('European call value')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'BCC_' + label + '.png')
    plt.savefig(save_path)
    plt.close()
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, call_diffs, label='call differences')
    plt2 = plt.plot(strikes, put_diffs, label='put differences')
    plt.title('Bates: Market Prices Minus Calibrated Prices ' + label)
    plt.xlabel('strike')
    plt.ylabel('calibration error')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'BCC_' + label + '_diff.png')
    plt.savefig(save_path)
    plt.close()


# Bates Model (with constant jumps) Calibration
def BCCJ_error_function(p0, strikes, exptimes, calls, puts, rates, y, const_jump):
    ''' Error Function for parameter calibration in BCC Model 

    Parameters as part of p0
    ==========
    sigma: float
        volatility factor in diffusion term
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    strikes: array of floats
        strike prices of the options
    T: float
        time to expiration
    calls: array of floats
        market call prices
    r: float
        risk-free interest rate
    y: float
        continuous dividend yield
    const_jump: float
        constant jump in v coinciding with spot
    Returns
    =======
    RMSE: float
        root mean squared error
    '''
    global i, min_RMSE
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta  = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or \
            rho < -1.0 or rho > 1.0 or delta < 0.0 or lamb < 0.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 500.0
    L = len(strikes)
    se = []
    for j in range(L):
        T = exptimes[j]
        S = S0*np.exp(-y*T)
        K = strikes[j]
        r = rates[j]
        call_model_value = BCCJ_call_value(S, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, const_jump)
        put_model_value = call_model_value - S + K * np.exp(-r * T)
        se.append((call_model_value - calls[j])**2 + (put_model_value - puts[j])**2)
    RMSE = np.sqrt(sum(se))
    #RMSE = np.sqrt(sum(se) / len(se))
    min_RMSE = min(min_RMSE, RMSE)
    if i % 50 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    i += 1
    return RMSE


def generate_plot_BCCJ(opt,strikes,calls,puts,T,r,y,const_jump,label):
    #
    # Calculating Model Prices
    #
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta  = opt
    
    call_model_values = []
    put_model_values = []
    call_diffs = []
    put_diffs = []
    L = len(strikes)
    S = S0*np.exp(-y*T)
    for j in range(L):
        K = strikes[j]
        call_model_values.append(BCCJ_call_value(S, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, const_jump))
        call_diffs.append(calls[j]-call_model_values[j])
        put_model_values.append(call_model_values[j] - S + K * np.exp(-r * T))
        put_diffs.append(puts[j]-put_model_values[j])
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, call_model_values, label='call model')
    plt2 = plt.plot(strikes, calls, label='call market')
    plt3 = plt.plot(strikes, put_model_values, label='put model')
    plt4 = plt.plot(strikes, puts, label='put market')
    plt.title('Bates (with a constant jump in v of 0.02): Market Prices Versus Calibrated Prices ' + label)
    plt.xlabel('strike')
    plt.ylabel('European call value')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'BCCJ_' + label + '.png')
    plt.savefig(save_path)
    plt.close()
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, call_diffs, label='call differences')
    plt2 = plt.plot(strikes, put_diffs, label='put differences')
    plt.title('Bates (with a constant jump in v of 0.02): Market Prices Minus Calibrated Prices ' + label)
    plt.xlabel('strike')
    plt.ylabel('calibration error')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'BCCJ_' + label + '_diff.png')
    plt.savefig(save_path)
    plt.close()


# Merton Jump Diffusion Calibration
def M76_error_function_analytic(p0,strikes,T,calls, puts, r,y):
    ''' Error Function for parameter calibration in M76 Model via
    analytical approach - one expiration date

    Parameters as part of p0
    ==========
    sigma: float
        volatility factor in diffusion term
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    strikes: array of floats
        strike prices of the options
    T: float
        time to expiration
    calls: array of floats
        market call prices
    r: float
        risk-free interest rate
    y: float
        continuous dividend yield

    Returns
    =======
    RMSE: float
        root mean squared error
    '''
    global i, min_RMSE
    sigma, lamb, mu, delta = p0
    if sigma < 0.0 or delta < 0.0 or lamb < 0.0:
        return 500.0
    L = len(strikes)
    se = []
    for j in range(L):
        T = exptimes[j]
        K = strikes[j]
        r = rates[j]
        call_model_value = M76_option_analytic(0,S0, K, T,
                                         r, y, sigma, lamb, mu, delta, 15)
        #put_model_value = call_model_value - S0 * np.exp(-y * T) + K * np.exp(-r * T)
        put_model_value = M76_option_analytic(1,S0, K, T, r, y, sigma, lamb, mu, delta, 15)
        se.append((call_model_value - calls[j])**2 + (put_model_value - puts[j])**2)
    RMSE = np.sqrt(sum(se))
    #RMSE = np.sqrt(sum(se) / len(se))
    min_RMSE = min(min_RMSE, RMSE)
    if i % 50 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    i += 1
    return RMSE


def generate_plot_M76(opt,strikes,calls,puts,T,r,y,label):
    #
    # Calculating Model Prices
    #
    sigma, lamb, mu, delta = opt
    call_model_values = []
    put_model_values = []
    call_diffs = []
    put_diffs = []
    L = len(strikes)
    for j in range(L):
        K = strikes[j]
        call_model_values.append(M76_option_analytic(0,S0, K, T, r, y, sigma, lamb, mu, delta, 15))
        call_diffs.append(calls[j]-call_model_values[j])
        put_model_values.append(M76_option_analytic(1,S0, K, T, r, y, sigma, lamb, mu, delta, 15))
        put_diffs.append(puts[j]-put_model_values[j])
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, call_model_values, label='call model')
    plt2 = plt.plot(strikes, calls, label='call market')
    plt3 = plt.plot(strikes, put_model_values, label='put model')
    plt4 = plt.plot(strikes, puts, label='put market')
    plt.title('Merton: Market Prices Versus Calibrated Prices ' + label)
    plt.xlabel('strike')
    plt.ylabel('European call value')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'M76_' + label + '.png')
    plt.savefig(save_path)
    plt.close()
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, call_diffs, label='call differences')
    plt2 = plt.plot(strikes, put_diffs, label='put differences')
    plt.title('Merton: Market Prices Minus Calibrated Prices ' + label)
    plt.xlabel('strike')
    plt.ylabel('calibration error')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'M76_' + label + '_diff.png')
    plt.savefig(save_path)
    plt.close()


# Heston Model Calibration

def H93_error_function(p0,strikes,exptimes,calls, puts, rates,y):
    ''' Error function for parameter calibration in BCC97 model via
    Lewis (2001) Fourier approach.

    Parameters
    ==========
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial, instantaneous variance

    Returns
    =======
    MSE: float
        mean squared error
    '''
    global i, min_MSE
    kappa_v, theta_v, sigma_v, rho, v0 = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or \
            rho < -1.0 or rho > 1.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 500.0
    L = len(strikes)
    se = []
    #S = S0*np.exp(-y*T)
    for j in range(L):
        T = exptimes[j]
        S = S0*np.exp(-y*T)
        K = strikes[j]
        r = rates[j]
        call_model_value = H93_call_value(S, K, T, r, kappa_v, theta_v, sigma_v, rho, v0)
        put_model_value = call_model_value - S + K * np.exp(-r * T)
        se.append((call_model_value - calls[j])**2 + (put_model_value - puts[j])** 2)
    #MSE = np.sqrt(sum(se) / len(se))
    MSE = np.sqrt(sum(se))
    min_MSE = min(min_MSE, MSE)
    if i % 25 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (MSE, min_MSE))
    i += 1
    return MSE


def generate_plot_H93(opt,strikes,calls,puts,T,r,y,label):
    #
    # Calculating Model Prices
    #
    kappa_v, theta_v, sigma_v, rho, v0 = opt
    call_model_values = []
    put_model_values = []
    call_diffs = []
    put_diffs = []
    L = len(strikes)
    S = S0*np.exp(-y*T)
    for j in range(L):
        K = strikes[j]
        call_model_values.append(H93_call_value(S, K, T, r, kappa_v, theta_v, sigma_v, rho, v0))
        call_diffs.append(calls[j]-call_model_values[j])
        put_model_values.append(call_model_values[j] - S + K * np.exp(-r * T))
        put_diffs.append(puts[j]-put_model_values[j])
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, call_model_values, label='call model')
    plt2 = plt.plot(strikes, calls, label='call market')
    plt3 = plt.plot(strikes, put_model_values, label='put model')
    plt4 = plt.plot(strikes, puts, label='put market')
    plt.title('Heston: Market Prices Versus Calibrated Prices ' + label)
    plt.xlabel('strike')
    plt.ylabel('European call value')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'H93_' + label + '.png')
    plt.savefig(save_path)
    plt.close()
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, call_diffs, label='call differences')
    plt2 = plt.plot(strikes, put_diffs, label='put differences')
    plt.title('Heston: Market Prices Minus Calibrated Prices ' + label)
    plt.xlabel('strike')
    plt.ylabel('calibration error')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'H93_' + label + '_diff.png')
    plt.savefig(save_path)
    plt.close()


# CIR Model Calibration
def CIR_error_function(p0,factors,times):
    ''' Error Function for parameter calibration in CIR Model via
    analytical approach - one discount factor per time

    Parameters as part of p0
    ==========
    r_0: float
        initial interest rate
    kappa_r: float
        mean reversion speed
    theta_r: float
        mean reversion level
    sigma_r: float
        volatility term
    factors: array of floats
        discount factors
    times: array of floats
        times to expiration for the factors

    Returns
    =======
    RMSE: float
        root mean squared error
    '''
    global i, min_RMSE
    r0, kappa_r, theta_r, sigma_r = p0

    if 2 * kappa_r * theta_r < sigma_r ** 2:
        return 100
    if kappa_r < 0 or theta_r < 0 or sigma_r < 0.001:
        return 100
    L = len(times)
    se = []
    for j in range(L):
        K = factors[j]
        alpha = [r0, kappa_r, theta_r, sigma_r,0, times[j]]
        model_value = B( alpha )
        se.append((model_value - K) ** 2)
    RMSE = np.sqrt(sum(se) / len(se))
    min_RMSE = min(min_RMSE, RMSE)
    if i % 50 == 0:
        print('%4d |' % i, np.array(p0), '| %10.6f | %10.6f' % (RMSE, min_RMSE))
    i += 1
    return RMSE

def generate_plotCIR(opt,factors,times):
    #
    # Calculating Model Prices
    #
    r0, kappa_r, theta_r, sigma_r = opt

    options = []
    L = len(factors)
    diffs = []
    for j in range(L):
        K = factors[j]
        alpha = [r0, kappa_r, theta_r, sigma_r, 0, times[j]]
        options.append(B(alpha))
        diffs.append(K-options[j])
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(times, options, label='model')
    plt2 = plt.plot(times, factors, label='market')
    plt.title('CIR: Market Curve Versus Calibrated Curve')
    plt.xlabel('time')
    plt.ylabel('discount factor')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'CIR.png')
    plt.savefig(save_path)
    plt.close()
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(times, diffs, label='differences')
    plt.title('CIR: Market Curve Minus Calibrated Curve')
    plt.xlabel('time')
    plt.ylabel('calibration error')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'CIR_diff.png')
    plt.savefig(save_path)
    plt.close()


# Plot Monte Carlo Results

def generate_plot_MC(strikes, american_option_values, market_prices, label, option):
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, american_option_values, label='model (american)')
    plt2 = plt.plot(strikes, market_prices, label='market (european)')
    plt.title('American {} Prices with Monte Carlo vs European Market Prices '.format(option) + label)
    plt.xlabel('Strike')
    plt.ylabel('American {} value'.format(option))
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'MC_american_' + "{}_{}".format(option,label) + '.png')
    plt.savefig(save_path)
    plt.close()
    market_price_diffs = []
    N = len(american_option_values)
    for j in range(N):
        market_price_diffs.append(american_option_values[j]-market_prices[j])
    plt.figure(figsize=(10, 6))
    plt1 = plt.plot(strikes, market_price_diffs, label='{} differences'.format(option))
    plt.title('American {} Prices with Monte Carlo minus European Market Prices '.format(option) + label)
    plt.xlabel('Strike')
    plt.ylabel('Price Difference')
    plt.grid(True)
    plt.legend(loc=4)
    plt.show(block=False)
    save_path = os.path.join(save_folder, 'MC_american_' + "{}_{}".format(option,label) + '_diff.png')
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    
    err = lambda y: np.sum((S0 * np.exp(-y * exptimes) - strikes * D_i - calls + puts) ** 2)
    err_deriv = lambda y: -2 * np.sum((S0 * np.exp(-y * exptimes) - strikes * D_i - calls + puts) * S0 * exptimes * np.exp(-y * exptimes))
    y_init = 0.00001
    tol = 0.0005
    approx_y = opt_y(err, err_deriv, y_init, tol, 20000)
    print("Dividend yield: ", round(approx_y,7))
    

    #
    # Bates Model Calibration
    #
    
    """ i = 0  # counter initialization
    min_RMSE = 100  # minimal RMSE initialization
    args = tuple()
    args = (strikes, exptimes, calls, puts, rates, approx_y)
    p0 = sop.brute(BCC_error_function,
            ((2.5, 10.6, 5.0),  # kappa_v
                (0.01, 0.041, 0.01),  # theta_v
                (0.05, 0.251, 0.1),  # sigma_v
                (-0.75, 0.01, 0.25),  # rho
                (0.01, 0.031, 0.01),  # v0
                (0.10, 0.401, 0.1),   # lambda
                (-0.5, 0.01, 0.1),   # mu
                (0.10, 0.301, 0.1)),   # delta
                args, finish=None)

    opt = sop.fmin(BCC_error_function, p0, args,
                maxiter=500, maxfun=750,
                xtol=0.000001, ftol=0.000001)
    """  
    opt=[1.12707126e+01,1.00202932e-02,1.97433702e-05,-7.56813636e-01,1.81378313e-02,1.97635655e-01,-3.19773472e-01,2.39109578e-01] #|  16.382 |  16.382

    # Split data into chunks to create plots of different maturities
    chunks = list(np.split(np.arange(25),5))
    for chunk in chunks:
        T = exptimes[chunk[0]]   # Just need one value of T for each group
        K = strikes[chunk]
        C = calls[chunk]
        P = puts[chunk]
        r = rates[chunk[0]]    # Just need one value of r for each group
        label = labels[chunk[0]]    # Just need one label for each group
        generate_plot_BCC(opt, K, C, P, T, r, approx_y, label)
    

    #
    # Bates Model (With Constant Jump) Calibration
    #
    
    const_jump = 0.02

    """ i = 0  # counter initialization
    min_RMSE = 100  # minimal RMSE initialization
    args = tuple()
    args = (strikes, exptimes, calls, puts, rates, approx_y, const_jump)
    p0 = sop.brute(BCCJ_error_function,
            ((2.5, 10.6, 5.0),  # kappa_v
                (0.01, 0.041, 0.01),  # theta_v
                (0.05, 0.251, 0.1),  # sigma_v
                (-0.75, 0.01, 0.25),  # rho
                (0.01, 0.031, 0.01),  # v0
                (0.10, 0.401, 0.1),   # lambda
                (-0.5, 0.01, 0.1),   # mu
                (0.10, 0.301, 0.1)),   # delta
                args, finish=None)

    opt = sop.fmin(BCCJ_error_function, p0, args,
                maxiter=500, maxfun=750,
                xtol=0.000001, ftol=0.000001)
      """
    opt = [1.81617688e+01,1.00846225e-02,5.97031368e-06,-3.42935226e-01,2.17483723e-02,2.21584422e-01,-2.81054430e-01, 2.43022126e-01]  #|  20.232 |  20.227

    # Split data into chunks to create plots of different maturities
    chunks = list(np.split(np.arange(25),5))
    for chunk in chunks:
        T = exptimes[chunk[0]]   # Just need one value of T for each group
        K = strikes[chunk]
        C = calls[chunk]
        P = puts[chunk]
        r = rates[chunk[0]]    # Just need one value of r for each group
        label = labels[chunk[0]]    # Just need one label for each group
        generate_plot_BCCJ(opt, K, C,P, T, r, approx_y, const_jump, label)
       

    #
    # Merton Jump Diffusion Calibration
    #
    
    """ i = 0  # counter initialization
    min_RMSE = 100  # minimal RMSE initialization
    args = tuple()
    args = (strikes, exptimes, calls, puts, rates, approx_y)
    p0 = sop.brute(M76_error_function_analytic,
                ((0.075, 0.201, 0.025),  # sigma
                    (0.10, 0.401, 0.1),  # lambda
                    (-0.5, 0.01, 0.1),  # mu
                    (0.10, 0.301, 0.1)),  # delta
                    args, finish=None)
    
    opt = sop.fmin(M76_error_function_analytic, p0, args,
                maxiter=500, maxfun=750,
                xtol=0.000001, ftol=0.000001)
     """
    opt = [ 0.11387867,  0.12602924, -0.44636368,  0.16678075] #|  28.121 |  28.121

    # Split data into chunks to create plots of different maturities
    chunks = list(np.split(np.arange(25),5))
    for chunk in chunks:
        T = exptimes[chunk[0]]   # Just need one value of T for each group
        K = strikes[chunk]
        C = calls[chunk]
        P = puts[chunk]
        r = rates[chunk[0]]   # Just need one value of r for each group
        label = labels[chunk[0]]   # Just need one label for each group
        generate_plot_M76(opt, K, C, P, T, r, approx_y, label)
    
    #
    # Heston Stochastic Volatility Calibration
    #

    """ i = 0  # counter initialization        
    min_MSE = 100  # minimal MSE initialization
    args = tuple()
    args = (strikes, exptimes, calls, puts, rates, approx_y)
    p0 = sop.brute(H93_error_function,
            ((2.5, 10.6, 5.0),  # kappa_v
                (0.01, 0.041, 0.01),  # theta_v
                (0.05, 0.251, 0.1),  # sigma_v
                (-0.75, 0.01, 0.25),  # rho
                (0.01, 0.031, 0.01)),  # v0
                args, finish=None)

    # second run with local, convex minimization
    # (dig deeper where promising)
    opt = sop.fmin(H93_error_function, p0, args,
            xtol=0.000001, ftol=0.000001,
            maxiter=750, maxfun=900)
     """ 
    opt = [ 1.64603105,  0.04335627,  0.37779825, -0.99999935,  0.01871849] #|  17.273 |  17.273

    # Split data into chunks to create plots of different maturities
    chunks = list(np.split(np.arange(25),5))
    for chunk in chunks:
        T = exptimes[chunk[0]]  # Just need one value of T for each group
        K = strikes[chunk]
        C = calls[chunk]
        P = puts[chunk]
        r = rates[chunk[0]]   # Just need one value of r for each group
        label = labels[chunk[0]]   # Just need one label for each group
        generate_plot_H93(opt, K, C, P, T, r, approx_y, label)
    
    
    #
    # CIR Calibration
    #

    rates_df = pd.read_excel(file_path, sheet_name='termrates')

    r_list = np.array(rates_df['termrates'])
    t_list = np.array(rates_df['exptimes'])
    factors = np.exp(-t_list * r_list)

    i = 0  # counter initialization
    min_RMSE = 100  # minimal RMSE initialization
    args = tuple()
    args = (factors,t_list)
    p0 = sop.brute(CIR_error_function,
                   ((0.0, 0.151, 0.05), # r0
                    (0.2, 10.3, 2.0), # kappa_r
                    (0.01, 0.11, 0.01), # theta_r
                    (0.05, 0.351, 0.1)), # sigma_r
                   args, finish=None)
    opt = sop.fmin(CIR_error_function, p0, args,
                   maxiter=500, maxfun=750,
                   xtol=0.000001, ftol=0.000001)
    
    opt = [0.05426532, 0.18368721, 0.00824399, 0.05503299]  #|   0.000075 |   0.000075
    generate_plotCIR(opt, factors, t_list)
     
    

    #
    # Monte Carlo American Option Calibration
    #

    american_df = pd.read_excel(file_path, sheet_name='american_valuation')
    exptimes = np.array(american_df["exptimes"])
    exptime = exptimes[0]
    strikes = np.array(american_df["strike"])
    calls = np.array(american_df["call_price"])
    puts = np.array(american_df["put_price"])
    rates = np.array(american_df["termrates"])
    rate = rates[0]
    labels = list(american_df["label"]) 
    label = labels[0] 

    opt_BCC = [ 1.81617688e+01, 1.00846225e-02, 5.97031368e-06, -3.42935226e-01,
                2.17483723e-02,  2.21584422e-01, -2.81054430e-01,  2.43022126e-01]
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta  = opt_BCC
    opt_CIR = [0.05426532, 0.18368721, 0.00824399, 0.05503299]
    r0, kappa_r, theta_r, sigma_r = opt_CIR
    #american_option_values = []
    option = ['call', 'put']

    for option_no, option_type in enumerate(option):
        american_option_values = []
        for strike in strikes:
            np.random.seed(100000)
            K = strike
            T = exptime
            M = 50
            I = 10000
            #r0 = 0.05426532
            S0 = 5123.41
            type = option_no
            #v0 = 2.10384669e-02
            y = approx_y # dividend yield
            #kappa_r = 0.18368721
            #theta_r = 0.00824399
            #sigma_r = 0.05503299
            #kappa_v = 1.81617688e+01
            #theta_v = 1.00846225e-02
            #sigma_v = 5.97031368e-06
            #lamb = 2.21584422e-01
            #mu = -2.81054430e-01
            #delta = 2.43022126e-01
            moment_matching = True
            anti_paths = True
            #rho = -7.02165782e-02 # correlation between variance process and index process
            rand = random_number_generator(M, I, anti_paths, moment_matching)
            cho_matrix = generate_cholesky(rho)
            r = SRD_generate_paths(r0, kappa_r, theta_r, sigma_r, T, M, I,
                                rand, 0, cho_matrix)
            v = SRD_generate_paths(v0, kappa_v, theta_v, sigma_v, T, M, I,
                                rand, 2, cho_matrix)
            S = B96_generate_paths(S0, r, y, v, lamb, mu, delta, rand, 1, 3,
                                cho_matrix, T, M, I, moment_matching)

            LSM_value = BCC97_lsm_valuation(type, S, r, v, K, T, M, I)
            american_option_values.append(LSM_value)
            print("LSM value %8.0f" % strike ,"=" , "%10.5f" % LSM_value )

        if option_type == 'call':
            market_prices = calls
        else:
            market_prices = puts
        
        # Generate plots for American Option
        generate_plot_MC(strikes, american_option_values, market_prices, label, option=option_type)
         
    
