import numpy as np
import matplotlib.pyplot as plt 

# Defining the relevant functions

T = 1
L = 150
S_0 = 100
r = 0.08
sigma_s = 0.30
sigma_v = 0.25
D = 175
corr = 0.2
recovery_rate = 0.25

def geometric_brownian_motion(N = 200, T = 1, S_0 = 1, mu=0.08, sigma = 0.05):
        """N is the number of steps, T is the time of expiry,
        S_0 is the beginning value, sigma is standard deviation"""
        deltat = float(T)/N #compute the size of the steps
        W = np.cumsum(np.random.standard_normal(size = N)) * np.sqrt(deltat)
        # generate the brownian motion
        t = np.linspace(0, T, N)
        X = (mu-0.5*sigma**2)*t + sigma*W
        S = S_0 * np.exp(X)
        # geometric brownian motion
        return S

def european_call_payoff(S_T, K = 1.5):
        """S_T is the price of the underlying at expiry.
        K is the strike price"""
        return np.maximum(0, S_T - K) #payoff for call option

def price_up_and_out_european_barrier_call(barrier, S_0, N = 500, K = 0.5):
        """barrier is the barrier level,
        S_0 is the starting value of the underlying,
        N is how many paths we will generate to take the mean.
        K is the strike price."""
        paths = [geometric_brownian_motion(S_0 = S_0) for i in range(N)]
        prices = []
        for path in paths:
                if np.max(path) > barrier: # knocked out
                        prices.append(0)
                else:
                        prices.append(european_call_payoff(path[-1], K)) 
        return np.mean(prices)


# Simulating the price depending on the current spot
# barrier is at 2.5
# strike price is at 0.5

spot = np.linspace(0, 3, 100)
prices = [price_up_and_out_european_barrier_call(2.5, s) for s in spot]
plt.plot(spot, prices, '--', linewidth=2)
plt.show()