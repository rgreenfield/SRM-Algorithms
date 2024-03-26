import numpy as np
import matplotlib.pyplot as plt

"""
Smoothing with Moving Averages
"""
    
# Generate some noisy data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

# Function to perform moving average smoothing
def moving_average(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data

# Apply moving average smoothing with a window size of 5
window_size = 5
smoothed_y = moving_average(y, window_size)

# Plot the original and smoothed data
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Noisy Data', color='lightgray', alpha=0.7)
plt.plot(x[window_size-1:], smoothed_y, label='Smoothed Data', color='blue')
plt.title('Smoothing with Moving Averages')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

"""
Double Smoothing with Moving Averages
"""


# Generate some synthetic data
np.random.seed(0)
x = np.arange(1, 101)
y = np.sin(x/10) + np.random.normal(0, 0.2, 100)

# Double exponential smoothing function
def double_exponential_smoothing(data, alpha, beta):
    smoothed_data = [data[0]]
    trend = [data[1] - data[0]]
    
    for i in range(1, len(data)):
        smoothed = alpha * data[i] + (1 - alpha) * (smoothed_data[-1] + trend[-1])
        trend_val = beta * (smoothed - smoothed_data[-1]) + (1 - beta) * trend[-1]
        smoothed_data.append(smoothed)
        trend.append(trend_val)
    
    return smoothed_data

# Apply double exponential smoothing
alpha = 0.2  # Smoothing factor for level
beta = 0.1   # Smoothing factor for trend
smoothed_y = double_exponential_smoothing(y, alpha, beta)

# Plot original and smoothed data
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Data', color='lightgray', alpha=0.7)
plt.plot(x, smoothed_y, label='Smoothed Data', color='blue')
plt.title('Double Exponential Smoothing with Moving Averages')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
