# SPX Index Option Pricing Model Calibration

* Working...
The goal of this project is to calibrate options pricing models to a
recent set of market data, including dividend rates, jumps, stochastic
volatility and interest rates, and use the result to price American options with
a Monte Carlo approach.

## Models Being Calibrated
* Bates (Jumps + Stochastic Volatility)
* Heston Stochastic Volatility
* Merton Jump Diffusion
* CIR Interest Rate Model
* American Option using Monte Carlo

## Market Data
We are working with S&P Index options for the following maturities: 6/21/2024,
9/20/2024, 12/20/2024, 3/21/2025 and 6/20/2025. There are 5 options for each
maturity. Prices are as of 04/12/2024. We will use the closing price for that day - 
$5,123.41 - as our index price in the models.

## 1. Calibrate an overal dividend yield

$$\begin{aligned} E^2 = \sum_{i=1}^{N} (S e^{-yT_{i}}- K_{i}D(T_{i}) - C_{i} + P_{i})^{2} \end{aligned}$$

where 𝑆, K_{i}, T_{i}, 𝑟_{i}, 𝐶_{i}, 𝑃_{i} are, respectively, current SPX index
price, strike price of the call-put pair for that strike price, expiration
date, interest rate to that expiration date, call and put price. Minimize E over
all values of y.

We used Newton's method to find:  

$$\begin{aligned} \frac{\partial E_{2}}{\partial y} = 0 \end{aligned}$$

The partial derivative is: 

$$\begin{aligned} \frac{\partial E_{2}}{\partial y} = -2 \sum_{i=1}^{N} (S e^{-yT_{i}} - K_{i}D(T_{i}) - C_{i} + P_{i}) \cdot S T_{i} e^{-yT_{i}} \end{aligned}$$

## 2. Calibrate the Bates model

$$\begin{aligned} \frac{dS}{S} = (r - y - \lambda (e^{\theta+\frac {1}{2} \beta^{2}} -1)) dt + \sqrt {v} dW_{1} + (e^{Y} - 1) dN \end{aligned}$$

$$\begin{aligned} dv = \kappa(\theta - v) dt + \alpha \sqrt {v} dW_{2} \end{aligned}$$

$$\begin{aligned} Y \sim \mathcal{N}(\theta, \beta^2) \end{aligned}$$

$$\begin{aligned} Q(dN = 1) = \lambda dt \end{aligned}$$

$$\begin{aligned} Q(dN = 0) = 1 - \lambda dt \end{aligned}$$

$$\begin{aligned} dW_{1} \cdot dW_{2} = \rho dt \end{aligned}$$

We calculate the least squares error of calibration:

$$\begin{aligned} E_{Bates} = \sqrt{\sum_{i=1}^{N} ((C_{i}^{\*} - C_{i})^{2}+(P_{i}^{\*} - P_{i})^{2})} \end{aligned}$$

### Bates Results
<img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_3m.png" width="500" height="auto">  

## 3. Calibrate the Heston Model

$$\begin{aligned} \frac{dS}{S} = (r - y) dt + \sqrt {v} dW_{1} \end{aligned}$$

$$\begin{aligned} dv = \kappa(\theta - v) dt + \alpha \sqrt {v} dW_{2} \end{aligned}$$

Just like with Bates, we will calculate the least squares error for Heston ($E_{Heston}$).

### Heston Results

## 4. Calibrate the Merton Model

### Merton Results

## 5. Calibrate the CIR Interest Rate Model

$$\begin{aligned} dr = \kappa_{r}(\theta_{r} - r)dt + \alpha_{r}rdW_{r} \end{aligned}$$

### CIR Results

## 6. Calibrate
