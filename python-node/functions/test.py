import numpy as np
import matplotlib.pyplot as plt

# ----- Fonctions et leurs dérivées -----
def f1(x):
    return (x-3)**2

def df1(x):
    return 2*(x-3)

def f2(x):
    return (x+1)**4  # une autre fonction

def df2(x):
    return 4*(x+1)**3

def f3(x):
    return np.sin(x) + 0.1*x**2  # fonction un peu chaotique

def df3(x):
    return np.cos(x) + 0.2*x

# ----- Descente de gradient -----
def gradient_descent(f, df, x0, lr=0.1, steps=30):
    x = x0
    history = [x]
    for _ in range(steps):
        x = x - lr * df(x)
        history.append(x)
    return history

# ----- Paramètres -----
x_range = np.linspace(-5, 5, 400)
start_points = [0, -4, 4]

# ----- Fonction pour tracer la descente -----
def plot_descent(f, df, x_range, x0, lr, steps, title):
    history = gradient_descent(f, df, x0, lr, steps)
    y = f(np.array(x_range))
    
    plt.figure(figsize=(8,5))
    plt.plot(x_range, y, label='Fonction', color='blue')
    plt.scatter(history, f(np.array(history)), color='red', zorder=5)
    plt.plot(history, f(np.array(history)), color='red', linestyle='--', alpha=0.5)
    
    for i, (hx, hy) in enumerate(zip(history, f(np.array(history)))):
        plt.text(hx, hy+0.5, f'{i}', color='black', fontsize=8)
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()

# ----- Exemples -----
plot_descent(f1, df1, x_range, x0=0, lr=0.2, steps=20, title='Descente de gradient sur f(x) = (x-3)^2')
plot_descent(f2, df2, x_range, x0=-4, lr=0.05, steps=30, title='Descente de gradient sur f(x) = (x+1)^4')
plot_descent(f3, df3, x_range, x0=4, lr=0.1, steps=25, title='Descente de gradient sur f(x) = sin(x) + 0.1x^2')
