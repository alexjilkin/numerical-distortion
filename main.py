import numpy as np
import math  
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.io.wavfile import read, write

# u is input x' = f(x, u)
N = 44000
width = 1
samplerate, data = read("clean.wav")
N = int(data.shape[0])

def odeEuler(f, x0, u):
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
    return ((u - x) / (4.2 * 10)) - ((2.52 * 0.2) * math.sinh(x / 45.3)) 

y = np.linspace(0, width, N)

u = np.array(data[:, 0], dtype=float)
normalize = np.vectorize(lambda y: y/14)
u = normalize(u)
print (u)

# u = (np.sin((np.pi * y * 2 * 100))) * 9

clip = np.vectorize(lambda y: 4.5 if y > 4.5 else -4.5 if y < -4.5 else y)
uClipped = clip(u)

x = odeEuler(circuit, 0.1, uClipped)

# plt.figure()
# plt.plot(y, uClipped, 'b')
# plt.plot(y, x, 'r')
# plt.axis([0, 0.05, -8, 8])
# plt.show()

write("clipped-ode-forwardeuler-2.wav", samplerate, x)
print("Created clipped-ode-forwardeuler.wav from clean.wav")
# NFFT=1024
# yf = rfft(uClipped)
# xf = rfftfreq(N, 1 / N)

# plt.plot(xf, np.abs(yf))
# plt.axis([0, 4000, -1000, 200000])
# plt.show()


