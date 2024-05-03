# SPX Index Option Pricing Model Calibration

* Under construction...

The goal of this project is to calibrate options pricing models to a
recent set of market data, including dividend rates, jumps, stochastic
volatility and interest rates, and use the result to price American options with
a Monte Carlo approach.

## Models Being Calibrated
* Bates (Jumps + Stochastic Volatility)
  * The code also includes a second version of the Bates model    
    with a constant jump in volatility coinciding with a jump in spot
* Heston Stochastic Volatility
* Merton Jump Diffusion
* CIR Interest Rate Model
* American Option using Monte Carlo

## Market Data
We are working with S&P Index options for the following maturities: 6/21/2024 (10 options),
9/20/2024 (10 options), 12/20/2024 (10 options), 3/21/2025 (9 options), 6/20/2025 (6 options), 
and 12/19/2025 (3 options). Prices are as of 04/12/2024. We will use the closing price for that day 
($5,123.41) as our index price in the models.

## 1. Calibrate an overal dividend yield

$$\begin{aligned} E^2 = \sum_{i=1}^{N} (S e^{-yT_{i}}- K_{i}D(T_{i}) - C_{i} + P_{i})^{2} \end{aligned}$$

where 𝑆, K_{i}, T_{i}, 𝑟_{i}, 𝐶_{i}, 𝑃_{i} are, respectively, current SPX index
price, strike price of the call-put pair for that strike price, expiration
date, interest rate to that expiration date, call and put price. Minimize E over
all values of y.

We used Newton's method to find:  

$$\begin{aligned} \frac{\partial E^{2}}{\partial y} = 0 \end{aligned}$$

The partial derivative is: 

$$\begin{aligned} \frac{\partial E^{2}}{\partial y} = -2 \sum_{i=1}^{N} (S e^{-yT_{i}} - K_{i}D(T_{i}) - C_{i} + P_{i}) \cdot S T_{i} e^{-yT_{i}} \end{aligned}$$

## 2. Error Calculation for Calibration of Options Models

We calculate the least squares error of calibration for each model:

$$\begin{aligned} E = \sqrt{\sum_{i=1}^{N} ((C_{i}^{\*} - C_{i})^{2}+(P_{i}^{\*} - P_{i})^{2})} \end{aligned}$$

The stars mean “theoretical” value, obtained with the calibrated parameters.

### Errors for each model

| Model                      | Error         |
| -------------------------- | ------------- |
| Bates                      | 26.669  |
| Bates (with constant jump  | 26.687  |
| Heston                     | 42.802  |
| Merton                     | 28.822  |

## 3. Calibrate the Bates model

$$\begin{aligned} \frac{dS}{S} = (r - y - \lambda (e^{\theta+\frac {1}{2} \beta^{2}} -1)) dt + \sqrt {v} dW_{1} + (e^{Y} - 1) dN \end{aligned}$$

$$\begin{aligned} dv = \kappa(\theta - v) dt + \alpha \sqrt {v} dW_{2} \end{aligned}$$

$$\begin{aligned} Y \sim \mathcal{N}(\theta, \beta^2) \end{aligned}$$

$$\begin{aligned} Q(dN = 1) = \lambda dt \end{aligned}$$

$$\begin{aligned} Q(dN = 0) = 1 - \lambda dt \end{aligned}$$

$$\begin{aligned} dW_{1} \cdot dW_{2} = \rho dt \end{aligned}$$

### Calibrated Parameters

| Parameter                  | Value         |
| -------------------------- | ------------- |
| kappa_v                      | 16.24  |
| theta_v  | 0.01  |
| sigma_v                     | 0.094  |
| rho                     | -0.43  |
| v0                     | 0.02  |
| lambda                     | 0.211  |
| mu                      | -0.312  |
| delta                     | 0.192  |

### Bates Results
<table>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_06-21-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_06-21-2024_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_09-20-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_09-20-2024_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_12-20-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_12-20-2024_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_03-21-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_03-21-2025_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_06-20-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_06-20-2025_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_12-19-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/BCC_12-19-2025_diff.png" width="450" height="auto"> 
    </td>
  </tr>
</table>

## 4. Calibrate the Heston Model

$$\begin{aligned} \frac{dS}{S} = (r - y) dt + \sqrt {v} dW_{1} \end{aligned}$$

$$\begin{aligned} dv = \kappa(\theta - v) dt + \alpha \sqrt {v} dW_{2} \end{aligned}$$

Just like with Bates, we will calculate the least squares error for Heston ($E_{Heston}$).

### Calibrated Parameter

| Parameter                  | Value         |
| -------------------------- | ------------- |
| kappa_v                      | 16.32  |
| theta_v  | 0.01  |
| sigma_v                     | 0.09  |
| rho                     | -0.44  |
| v0                     | 0.02  |

### Heston Results

<table>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_06-21-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_06-21-2024_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_09-20-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_09-20-2024_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_12-20-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_12-20-2024_diff.png" width="450" height="auto">  
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_03-21-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_03-21-2025_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_06-20-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_06-20-2025_diff.png" width="450" height="auto">  
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_12-19-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/H93_12-19-2025_diff.png" width="450" height="auto">  
    </td>
  </tr>
</table>

## 5. Calibrate the Merton Model

$$\begin{aligned} \sum_{n=0}^{\infty} \frac{e^{-\tilde{\lambda} T}(\tilde{\lambda} T)^{n}}{n!} (S_{0} e^{-yT}N(d_{1,n}) - Ke^{-r_{n}T}N(d_{2,n})) \end{aligned}$$

$$\begin{aligned} d_{1,n} = \frac{ln(\frac{S_{0}}{K}) + (r_{n} - y + \frac{1}{2} \sigma_{n}^{2})T}{\sigma_{n} \sqrt{T}} \end{aligned}$$

$$\begin{aligned} d_{2,n} = d_{1,n} - \sigma_{n} \sqrt{T} \end{aligned}$$

### Calibrated Parameter

| Parameter                  | Value         |
| -------------------------- | ------------- |
| sigma                     | 0.112  |
| lambda                    | 0.155  |
| mu                         | -0.382  |
| delta                      | 0.16  |

### Merton Results

<table>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_06-21-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_06-21-2024_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_09-20-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_09-20-2024_diff.png" width="450" height="auto">  
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_12-20-2024.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_12-20-2024_diff.png" width="450" height="auto">  
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_03-21-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_03-21-2025_diff.png" width="450" height="auto"> 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_06-20-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_06-20-2025_diff.png" width="450" height="auto">  
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_12-19-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/M76_12-19-2025_diff.png" width="450" height="auto">  
    </td>
  </tr>
</table>

## 6. Calibrate the CIR Interest Rate Model

$$\begin{aligned} dr = \kappa_{r}(\theta_{r} - r)dt + \alpha_{r}rdW_{r} \end{aligned}$$

### CIR Results

<table>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/CIR.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/CIR_diff.png" width="450" height="auto"> 
    </td>
   </tr>   
</table>

## 7. Compute Value of American Option Using Monte Carlo
We put everything together for a 3-factor model

$$\begin{aligned} \frac{dS}{S} = (r - y - \lambda (e^{\theta+\frac {1}{2} \beta^{2}} -1)) dt + \sqrt {v} dW_{1} + (e^{Y} - 1) dN \end{aligned}$$

$$\begin{aligned} dv = \kappa(\theta - v) dt + \alpha \sqrt {v} dW_{2} \end{aligned}$$

$$\begin{aligned} Y \sim \mathcal{N}(\theta, \beta^2) \end{aligned}$$

$$\begin{aligned} Q(dN = 1) = \lambda dt \end{aligned}$$

$$\begin{aligned} Q(dN = 0) = 1 - \lambda dt \end{aligned}$$

$$\begin{aligned} dW_{1} \cdot dW_{2} = \rho dt \end{aligned}$$

$$\begin{aligned} dr = \kappa_{r}(\theta_{r} - r)dt + \alpha_{r}rdW_{r} \end{aligned}$$

$$\begin{aligned} dW_{r} \cdot dW_{1} = dW_{r} \cdot dW_{2} = 0 \end{aligned}$$

<table>
  <tr>
    <td>
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/MC_american_put_03-21-2025.png" width="450" height="auto">  
      <img src="https://github.com/josephobonyo/option_calibration/blob/main/graphs/MC_american_put_03-21-2025_diff.png" width="450" height="auto"> 
    </td>
  </tr>
</table>
