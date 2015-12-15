import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import cos
from math import exp

# u_t = u_xx , -1<x<1, 0<t<0.5
mu = 0.4
h = 1./20 # 20 40 80
grid_num = int(2./h)
k = mu*h*h
sigma_to = 100 # used to calculate exact solution. ideally, inf.

#number of v
nv = int(2/h)

v = np.zeros(grid_num+1)
#initial condition
for k in range(grid_num+1):
    if -1+k*h < -1./2 or -1+k*h > 1./2 :
        v[k] = 0
    elif abs(-1+k*h) == 1./2 :
        v[k] = 1./4
    else:
        if -1+k*h > 0:
            v[k] = 2-k*h
        else:
            v[k] = k*h

def sigma(limit, func):
    z = 0
    for i in range(limit):
        z += func(i)
    return z

def exact_solution(t, x) :
    return 3./8 + sigma(sigma_to, lambda l: (pow(-1, l)/(pi * (2*l+1)) + 2/(pi**2 * (2*l + 1)**2 ))*cos(pi*(2*l+1)*x)*exp(-pi**2*(2*l+1)**2*t)) + \
           sigma(sigma_to, lambda m: cos(2*pi*(2*m + 1)*x) / (pi**2*(2*m+1)**2 ) * exp(-4* pi**2 *(2*m+1)**2*t))

