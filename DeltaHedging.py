import numpy as np
import scipy.stats as stats

#Black Scholes Merton
def calculate_d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

def bsm_Call(S, K, r, sigma, T):
    d1 = calculate_d1(S, K, r, sigma, T)
    d2 = (np.log(S / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    c = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return c


#Delta Hedging
def deltaHedging(S, K, r, sigma, T, n):
    delta_t = T/n
    z = stats.norm.rvs(size = n) #random numbers
    
    t_left = T             #Time to expiration
    stock = S                  #stock price
    delta = [stats.norm.cdf(calculate_d1(S, K, r, sigma, T))]          #Delta

    boughtStocks = delta[0]
    cost = boughtStocks * stock
    kum = cost
    interest = kum * np.exp(delta_t * r) - kum

    for i in range(1, n-1):
        t_left = delta_t * (n - i)
        stock = stock * np.exp(delta_t * (r - 0.5 * sigma**2) + np.sqrt(delta_t) * sigma * z[i])
        d1 = calculate_d1(stock, K, r, sigma, t_left)
        delta.append(stats.norm.cdf(d1))
        boughtStocks = delta[i] - delta[i-1]
        cost = boughtStocks * stock
        kum = kum + cost + interest                #Total Borrowing
        interest = kum * np.exp(delta_t * r) - kum

    #for last time step, because otherwise t_left = 0, can't divide by 0
    t_left = 0.00001
    stock = stock * np.exp(delta_t * (r - 0.5 * sigma**2) + np.sqrt(delta_t) * sigma * z[n-1])
    d1 = calculate_d1(stock, K, r, sigma, t_left)
    delta.append(stats.norm.cdf(d1))
    boughtStocks = delta[n-1] - delta[n-2]
    cost = boughtStocks * stock
    kum = kum + cost + interest                #Total Borrowing
    interest = kum * np.exp(delta_t * r) - kum

    if(stock > K):
        hedgingcost = kum - K
    else:
        hedgingcost = kum
        
    return hedgingcost / (np.exp(T * r)) #PV
   
    
#Assumptions
S = 49                 #spot price
K = 50                 #strike price
sigma = 0.2            #volatility
T = 10/52              #10 weeks
r = 0.05               #risk free
n = 20000              #steps

print("Black-Scholes-Price = ", bsm_Call(S, K, r, sigma, T))
print("Hedging-Cost = ", deltaHedging(S, K, r, sigma, T, n))
