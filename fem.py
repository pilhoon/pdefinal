import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from math import pi
from math import cos
from math import exp
import pdb
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dtinv', type=int, default=400, help='inverse of dt') 
parser.add_argument('--dxinv', type=int, default=20, help='inverse of dx') 
args = parser.parse_args()

dx = 1./args.dxinv # 10 20 40 80 , dx = h
x_coords = np.linspace(-1, 1, 2./dx + 1)
dt = 1./args.dtinv
t_coords = np.linspace(0, 0.5, 0.5/dt + 1)


def sigma(limit, func):
    z = 0
    for i in range(limit):
        z += func(i)
    return z

def exact_solution(t, x) :
    sigma_to = 100 # used to calculate exact solution. ideally, inf.
    return 3./8 + sigma(sigma_to, lambda l: (pow(-1, l)/(pi * (2*l+1)) + 2/(pi**2 * (2*l + 1)**2 ))*cos(pi*(2*l+1)*x)*exp(-pi**2*(2*l+1)**2*t)) + \
           sigma(sigma_to, lambda m: cos(2*pi*(2*m + 1)*x) / (pi**2*(2*m+1)**2 ) * exp(-4* pi**2 *(2*m+1)**2*t))


def initc():
    c = []
    global x_coords
    for x in x_coords:
        if x < -1./2 or x > 1./2 :
            c = np.append(c, 0)
        elif abs(x) == 1./2:
            c = np.append(c, 1./4)
        else:
            c = np.append(c, 1-abs(x))
    return c


def main(dx, dt):
    # matrix M
    global x_coords, t_coords
    d = len(x_coords)
    M = K = np.zeros((d,d))
    for i in range(d): #diagonal
        M[i,i] = 4.*dx/6 
        K[i,i] = 2./dx
    for i in range(1,d): #upper and lower diagonal
        M[i, i-1] = M[i-1, i] = 1.*dx/6
        K[i, i-1] = K[i-1, i] = -1./dx

    Minv = np.linalg.inv(M)
    MinvK = np.dot(Minv, K)
    #c = np.reshape(initc(), (d, 1))
    c = initc()
    for t in t_coords:
        #pdb.set_trace()
        c = np.subtract(c, np.dot(MinvK, c*dt))

    plt.plot(c)
    plt.show()

main(dx, dt)
