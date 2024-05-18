import numpy as np
import matplotlib.pyplot as plt

# Global variables
n = 20
w = 0.5
y = [37, 28, 75,29,	31,	20,	52,	69,	72,	59,	30,	34,	86,	73,	86,	40,	71,	58,	46,	45,	51,]
t = np.array([i for i in range(len(y))])
l = [i for i in range(5)]

w_init = 0.5
learning_rate = 0.01
iterations = 100

def exponential_smoothing_1(w, y):
    s_t = []
    
    for i in range(len(y)):
        if i == 0:
            s_t.append((1 - w) * y[i])
        else:
            s_t.append((1 - w) * y[i] + w * s_t[i - 1])

    return np.array(s_t)

def exponential_smoothing_2(w, s_t):    
    s_t_2 = []
    
    for i in range(len(s_t)):
        if i == 0:
            s_t_2.append((1 - w) * s_t[i])
        else:
            s_t_2.append((1 - w) * s_t[i] + w * s_t_2[i - 1])
    
    return np.array(s_t_2)

def SS_w(y, s_t):
    
    ss_w = []
    
    for i in range(1, len(y)):
        ss_w.append((y[i] - s_t[i-1])**2)
    
    print(np.round(sum(ss_w)))
    
def gradient_descent(y, w_init, learning_rate, iterations):
    w = w_init
    
    for i in range(iterations):
        s_t = exponential_smoothing_1(w, y)
        loss = SS_w(y, s_t)
        
        # Calculate the gradient (derivative of the loss with respect to w)
        gradient = np.sum([-2 * (y[j] - s_t[j-1]) * (y[j] - s_t[j-1] + (1 - w) * y[j-1]) for j in range(1, len(y))])
        
        # Update w
        w = w - learning_rate * gradient
        
        if i % 10 == 0:
            print(f"Iteration {i}: w = {w}, loss = {loss}")
        
    return w
        
def calc_b_0(n, s_t, s_t_2):
    """_summary_
    Args:
        n (int): value of t forcasting from
        s_t (list or array): exponential smoothing
        s_t_2 (list or array): Double exponential smoothing
    """

    b_0 = (2*s_t[n] - s_t_2[n])
    return b_0

def calc_b_1(n, w, s_t, s_t_2):
    """_summary_
    Args:
        n (int): value of t forcasting from
        s_t (list or array): exponential smoothing
        s_t_2 (list or array): Double exponential smoothing
    """
    
    b_1 = ((1 - w) / w) * (s_t[n] - s_t_2[n])
    return b_1

def y_pred(b0, b1, l):
    
    t = [i for i in range(len(y), len(y) + len(l))]
    
    y_hat =[]
    
    for i in range(1, len(l)+1):
        y_hat.append(b0 + b1*l[i-1])
        
    return y_hat
    
def plot_t_y_st(t, y, s_t, s_t_2):
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(t, y, marker='o', linestyle='-', color='b', label='y(t)')
    plt.plot(t, s_t, marker='o', linestyle='--', color='g', label='s^1(t)')
    plt.plot(t, s_t_2, marker='o', linestyle='dashdot', color='y', label='s^2(t)')

    # Add labels and title
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Plot of t vs. y')
    plt.legend()
    plt.show()

def main():
    
    s_t = exponential_smoothing_1(w, y)
    s_t_2 = exponential_smoothing_2(w, s_t)
    
    print(s_t_2)
    print(len(s_t_2))
    SS_w(y, s_t)
    
    b0 = calc_b_0(n, s_t, s_t_2)
    b1 = calc_b_0(n, s_t, s_t_2)
    y_hat = y_pred(b0, b1, l)
    print(y_hat)
    #plot_t_y_st(t, y, s_t, s_t_2)    
    
    #optimal_w = gradient_descent(y, w_init, learning_rate, iterations)
    #git statprint(f"Optimal w: {optimal_w}")
    
    
if __name__ == '__main__':
    main()

