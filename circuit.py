import math

R = 2.2

def circuit(x, u): 
    return ((u - x) / (R * 10)) - ((0.504) * math.sinh(x / 45.3)) 