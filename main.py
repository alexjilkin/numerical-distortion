import numpy as np
import math  
import matplotlib.pyplot as plt
# from scipy.fft import rfft, rfftfreq
from scipy.io.wavfile import read, write
import sys

# u is input x' = f(x, u)

file_name = "clean"
print(file_name)
width = 1
sample_rate, data = read("{0}.wav".format(file_name))

# Length of input
N = int(data.shape[0])
R = 4.4

def odeBackwardEuler(f, x0, u):
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


def odeForwardEuler(f, x0, u):
    '''
        x0 - initial value
        f - f(x, u) 
    '''
    x = np.zeros(N)
    x[0] = x0

    for n in range(0, N - 1):
        x[n+1] = x[n] + (f(x[n], u[n]) * (1))
    return x

def circuit(x, u): 
    return ((u - x) / (R * 10)) - ((0.504) * math.sinh(x / 45.3)) 

y = np.linspace(0, width, N)

u = np.array(data[:, 0], dtype=float)
normalize = np.vectorize(lambda y: y / (400))
u = normalize(u)

#u = (np.sin((np.pi * y * 2 * 100))) * 9
reduceVolume = np.vectorize(lambda y: y / (8))

clip = np.vectorize(lambda y: 4.5 if y > 4.5 else -4.5 if y < -4.5 else y)
uClipped = clip(u)

# xb = odeBackwardEuler(circuit, 0.1, uClipped)
# xb = reduceVolume(xb)

# write("clipped-ode-backward-difference-{0}.wav".format(file_name), sample_rate, xb)
# print("Created clipped-ode-backward-difference-{0}.wav from {0}".format(file_name))

xf = odeForwardEuler(circuit, 0.1, uClipped)

xf = reduceVolume(xf)

write("forward-difference-{0}-R={1}.wav".format(file_name, R), sample_rate, xf)
print("Created forward-difference-{0}.wav from {0}".format(file_name))
# plt.figure()
# plt.plot(y, uClipped, 'b')
# plt.plot(y, xb, 'r')
# plt.plot(y, xf, 'y:')
# plt.axis([0, 0.05, -8, 8])
# plt.show()


# NFFT=1024
# yf = rfft(uClipped)
# xf = rfftfreq(N, 1 / N)

# plt.plot(xf, np.abs(yf))
# plt.axis([0, 4000, -1000, 200000])
# plt.show()


