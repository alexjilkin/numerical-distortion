import numpy as np
import math  
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import sys
from numericalMethods import odeBackwardEuler, odeForwardEuler
from circuit import  circuit

file_name = "samples/clean"
width = 1
sample_rate, data = read("{0}.wav".format(file_name))

# Length of input
N = int(data.shape[0])

y = np.linspace(0, width, N)

u = np.array(data[:, 0], dtype=float)
normalize = np.vectorize(lambda y: y / (400))
u = normalize(u)

reduceVolume = np.vectorize(lambda y: y / (8))

clip = np.vectorize(lambda y: 4.5 if y > 4.5 else -4.5 if y < -4.5 else y)
uClipped = clip(u)

xf = odeForwardEuler(circuit, 0.1, uClipped, N)

xf = reduceVolume(xf)

write("{0}-forward-differenc.wav".format(file_name), sample_rate, xf)
print("Created {0}-forward-differenc.wav from {0}".format(file_name))
