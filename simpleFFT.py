import numpy as np
from scipy.fftpack import fft
import timeit



def M76_characteristic_function(u, T, r, q, sigma, lamb, mu, delta):
    ''' Characteristic function for Merton jump diffusion'''
    omega = r - q - 0.5 * sigma ** 2 - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    value = np.exp((1j * u * omega - 0.5 * u ** 2 * sigma ** 2 +
            lamb * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1)) * T)
    return value


def exp_of_xTtrap(T, S0, K, paramvec, n):

    x1, dens = characteristic_function_to_density(2**n,np.log(K/S0)-np.log(np.log(2**n)),np.log(K/S0)+np.log(np.log(2**n)),T,paramvec)
    
    y = np.exp(x1)
    
    dy = np.diff(y)
    
    kstrike = 2**(n-1)
    
    fterms = ((y[kstrike:-1]- y[kstrike])*dens[kstrike:-1]+(y[kstrike+1::]-y[kstrike])*dens[kstrike+1::])/2
    int_value = dy[kstrike::].dot(fterms)
    call_value = S0*np.exp(-paramvec[0] * T) * int_value
    
    return call_value


def characteristic_function_to_density(
  N,   # Number of points, ideally a power of 2
  lna, lnb, # Evaluate the density on [a,b]
  T,
  paramVec):
  
  a = lna
  b = lnb
  i=np.arange(0, N , 1)  # Indices
  dx = (b-a)/N           # Step size, for the density
  x = a + i * dx         # Grid, for the density
  du = 2*np.pi / ( N * dx ) # Step size, frequency space
  c = -N/2 * du          # Evaluate the characteristic function on [c,d]
  u = c + i * du         # Grid, frequency space
  phi_u = M76_characteristic_function(u, T, paramVec[0], paramVec[1], paramVec[2], paramVec[3], paramVec[4], paramVec[5])
  X = np.exp(-1j* i * du * a ) * phi_u
  Y = fft(X)
  density = (du / (2*np.pi*np.exp(x)) * np.exp(-1j * c * x ) * Y).real
  
  return x, density


if __name__ == '__main__':
    
    # M76
    p0=np.array([0.111, 2.066, -0.124, 0.090])
    Spot = 3830.17
    r = 0.00138134 #risk-free rate
    q = 0.01609341 #dividend yield
    K = 3976.381 #strike price
    #2/3/2021
    #Strike    delta
    #3585.251 #80 delta 
    #3698.364 #70 delta 
    #3773.699 #60 delta 
    #3803.889 #55 delta 
    #3830.939 #50 delta 
    #3855.871 #45 delta 
    #3879.493 #40 delta 
    #3925.557 #30 delta 
    #3976.381 #20 delta
    T = 30/365
    #S0 = Spot*np.exp(-q*T)
    S0 = Spot
    sigma = p0[0]
    lamb = p0[1]
    mu = p0[2]
    delt = p0[3]
    params = [r, q, sigma, lamb, mu, delt]

    start0 = timeit.default_timer()
    cp0=exp_of_xTtrap(T, S0, K, params, 15)
    stop0 = timeit.default_timer()
    print("Value of Call Option T0 %8.4f"% cp0)
    print('Time: ', stop0 - start0)