import numpy as np
import math  
import matplotlib.pyplot as plt

def odeForwardEuler(f, x0, u, N):
    '''
        x0 - initial value
        f - f(x, u) 
    '''
    x = np.zeros(N)
    x[0] = x0

    for n in range(0, N - 1):
        x[n+1] = x[n] + (f(x[n], u[n]) * (1))
    return x

def odeBackwardEuler(f, x0, u, N):
    ''' x0 - initial value f - f(x, u) '''
    x = np.zeros(N)
    x[0] = x0

    for n in range(0, N - 1):
        # Initial guess using forward euler
        i_g = x[n] + (f(x[n], u[n]) * (1))

        for _ in range(0, 10):
            gx = i_g - x[n] - f(i_g, u[n+1]) 
            gdx = ((0.504 / 45.3) * math.cosh(i_g / 45.3)) + (1 / (R * 10)) + 1
            i_g = i_g - (gx / gdx)

        x[n+1] = i_g
    return x


