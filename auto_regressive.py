import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Parameters
n = 100  # Length of the time series
phi = 0.6  # Autoregressive coefficient (AR(1))

# Generate AR(1) time series
np.random.seed(0)
ar = [0]  # Initial value
for i in range(1, n):
    ar.append(phi * ar[i-1] + np.random.normal())

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(range(n), ar, label=f'AR(1) with φ={phi}')
plt.title('Autoregressive (AR(1)) Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()