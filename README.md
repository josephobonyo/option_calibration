# SPX Index Option Pricing Model Calibration

* Working...

## Models Being Calibrated
* Bates (Jumps + Stochastic Volatility)
* Merton Jump Diffusion
* Hestons Stochastic Volatility

## 1. Calibrate an overal dividend yield

$\ E^2 = sum_{i=1}^{N} (S e^{-yT_{i}}- K_{i}D(T_{i}) - C_{i} + P_{i})^{2} $  

where 𝑆, K_{i}, T_{i}, 𝑟_{i}, 𝐶_{i}, 𝑃_{i} are, respectively, current SPX index
price, strike price of the call-put pair for that strike price, expiration
date, interest rate to that expiration date, call and put price. Minimize E over
all values of y.

We used Newton's method to find:
$\frac{\partial E_{2}}{\partial y} = 0$  

The partial derivative is:
$\frac{\partial E_{2}}{\partial y} = -2 \sum_{i=1}^{N} (S e^{-yT_{i}} - K_{i}D(T_{i}) - C_{i} + P_{i}) $  

## 2. Calibrate the Bates model

$\frac{dS}{S} = r - y - \lambda (e^{\theta+\frac {1}{2} \beta^{2}} -1) dt + \sqrt {v} dW_{1} + (e^{Y} - 1) dN $  

$\ dv = \kappa(\theta - v) dt + \alpha v dW_{2} $  

$\ Y \sim \mathcal{N}(\theta, \beta^2) $  

$\ Q(dN = 1) = \lambda dt $  

$\ Q(dN = 0) = 1 - \lambda dt $  

$\ dW_{1} \cdot dW_{2} = \rho dt $