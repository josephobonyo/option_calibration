#
# Valuation of European call options in Black-Scholes-Merton model
# incl. Vega function and implied volatility estimation
# This version includes dividend yield, as in Garman-Kohlhagen
# bsm_functions_GK.py
#

# Analytical Black-Scholes-Merton (BSM) Formula


def bsm_call_value_GK(S0, K, T, r, y, sigma):
    ''' Valuation of European call option in BSM model.
    Analytical formula.
    
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    y : float
        constant dividend yield
    sigma : float
        volatility factor in diffusion term
    
    Returns
    =======
    value : float
        present value of the European call option
    '''
    from math import log, sqrt, exp
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r - y + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - y - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * exp(-y * T) * stats.norm.cdf(d1, 0.0, 1.0)
            - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
      # stats.norm.cdf --> cumulative distribution function
      #                    for normal distribution
    return value

# Vega function


def bsm_vega_GK(S0, K, T, r, y, sigma):
    ''' Vega of European option in BSM Model.
    
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    y : float
        constant dividend yield
    sigma : float
        volatility factor in diffusion term
    
    Returns
    =======
    vega : float
        partial derivative of BSM formula with respect
        to sigma, i.e. Vega

    '''
    from math import log, sqrt, exp
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r - y + (0.5 * sigma ** 2) * T)) / (sigma * sqrt(T))
    vega = S0 * exp( -y * T ) * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)
    return vega

# Implied volatility function


def bsm_call_imp_vol_GK(S0, K, T, r, y, C0, sigma_est, it=100):
    ''' Implied Volatility of European call option in BSM Model.
    
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    y : float
        constant dividend yield
    sigma_est : float
        estimate of impl. volatility
    it : integer
        number of iterations
    
    Returns
    =======
    sigma_est : float
        numerically estimated implied volatility
    '''
    for i in range(it):
        sigma_est -= ((bsm_call_value_GK(S0, K, T, r, y, sigma_est) - C0)
                        / bsm_vega_GK(S0, K, T, r, y, sigma_est))
    return sigma_est
