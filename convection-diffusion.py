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

parser = argparse.ArgumentParser(add_help= False)
parser.add_argument('-a', type=float, default=1)
parser.add_argument('-b', type=float, default=0.1)
parser.add_argument('-c', type=float, default=1)
parser.add_argument('-k', type=int, default=400, help='inverse of dt') 
parser.add_argument('-h', type=int, default=20, help='inverse of dx') 
args = parser.parse_args()

dx = 1./args.h
x_coords = np.linspace(-10, 10, 2./dx + 1)
dt = 1./args.k
t_coords = np.linspace(0, 10, 1./dt + 1)
a = args.a
b = args.b
c = args.c
# mu = dt / (dx*dx)
# lmbda = dt / dx

def exact_solution(t, x, a, b, c): #x : array
    k = np.multiply(1.*c/(2*b), np.subtract(x, a*t))
    return np.subtract(a, np.multiply(c, np.tanh(k)))

def yinit():
    return exact_solution(0, x_coords, a, b, c)

def ftcs1_step(y): # 1/2 * (u^2)_x = u*u_x
    global dt, dx, a, b, c
    y_1 = np.append(y[1], y)[:-1]
    y1 = np.append(y, y[-2])[1:]
    terms = [ np.multiply(dt*b /(dx**2), y1), \
              np.multiply(-2*dt*b /(dx**2) + 1, y), \
              np.multiply(dt*b/(dx**2), y_1), \
              np.multiply(-dt/(2*dx), np.multiply(y, y1)), \
              np.multiply(dt/(2*dx), np.multiply(y, y_1))]
    return np.sum(terms, axis=0)
    
def ftcs2_step(y):
    global dt, dx, a, b, c
    y_1 = np.append(y[1], y)[:-1]
    y1 = np.append(y, y[-2])[1:]
    terms = [ np.multiply(dt*b /(dx**2), y1), \
              np.multiply(-2*dt*b /(dx**2) + 1, y), \
              np.multiply(dt*b/(dx**2), y_1), \
              np.multiply(-dt/(4*dx), np.power(y1, 2)), \
              np.multiply(dt/(4*dx), np.power(y_1, 2))]
    return np.sum(terms, axis=0)

#
# exact solution view
#
#fig, ax = plt.subplots()
#y= exact_solution(0, x_coords, a, b, c)
#line, = ax.plot(x_coords,y)
##plt.plot(x_coords, exact_solution(0.5, x_coords, 1,1,1))
#def update(t):
#    print t
#    line.set_ydata(exact_solution(t, x_coords, a, b, c))
#    return line
#    
#ani = ani.FuncAnimation(fig, update, t_coords, interval=1, repeat=False)
#plt.show()

#
# ftcs view
#
#fig, ax = plt.subplots()
#y= exact_solution(0, x_coords, a, b, c)
#line, = ax.plot(x_coords,y)
###plt.plot(x_coords, exact_solution(0.5, x_coords, 1,1,1))
#def update(t):
#    global y
#    y = ftcs2_step(y)
#    line.set_ydata(y)
#    return line
##    
#ani = ani.FuncAnimation(fig, update, t_coords, interval=25, repeat=False)
#plt.show()





