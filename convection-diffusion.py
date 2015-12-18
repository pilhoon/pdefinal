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
parser.add_argument('--type', type=int, default=1) 
parser.add_argument('--animate', action='store_true')  #if not show error
parser.add_argument('--interval', type=int , default=25)  #if not show error
parser.add_argument('--show', action='store_true', help='show error graph')
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--xmin', type=int, default=-10)
parser.add_argument('--xmax', type=int, default=10)
parser.add_argument('--ymin', type=int, default=-10)
parser.add_argument('--ymax', type=int, default=10)
parser.add_argument('--solution', action='store_true')
args = parser.parse_args()

assert args.xmax > args.xmin 

dx = 1.*(args.xmax - args.xmin)/args.h
x_coords = np.linspace(args.xmin, args.xmax, args.h + 1)
dt = 1.*args.tmax/args.k
t_coords = np.linspace(0, args.tmax, args.k + 1)
a = args.a
b = args.b
c = args.c
# mu = dt / (dx*dx)
# lmbda = dt / dx

print '---------------------'
print ' a = ', a
print ' b = ', b
print ' c = ', c
print ' dx = ', dx, '(h = ', args.h, ')'
print ' dt = ', dt, '(k = ', args.k, ')'
print ' t = 0 ~', args.tmax, '(len = ', len(t_coords), ')'
print ' x = ', args.xmin, '~', args.xmax, '(len = ', len(x_coords), ')'
print ' dx*c/2b = ', (dx*c)/(2*b)

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

def get_error(type=args.type):
    global a, b, c, t_coords, x_coords, dx, dt
    y = yinit()
    err = []
    for idx , t in enumerate(t_coords[1:]):
        if type == 1 :
            y = ftcs1_step(y)
        else:
            y = ftcs2_step(y)
        err_arr = y - exact_solution(t, x_coords, a, b, c)
        norm = linalg.norm(err_arr)
        err += [norm]
        if norm> 20:
            print ' \033[31mdiverge at ', idx , '/', len(t_coords)-1, '\033[0m'
            break;
    print ' error avg = ', np.average(err)
    print '       std = ', np.std(err)
    print '       max = ', max(err)
    print '       min = ', min(err)
    return err

#
# exact solution view
#

if args.animate :
    #
    # ftcs view
    #
    print ' interval = ', args.interval , ' ms'
    fig, ax = plt.subplots()
    ax.set_xlim([args.xmin, args.xmax])
    ax.set_ylim([args.ymin, args.ymax])
    y = exact_solution(0, x_coords, a, b, c)
    line, = ax.plot(x_coords,y)
    ##plt.plot(x_coords, exact_solution(0.5, x_coords, 1,1,1))
    def update(t):
        print '\r @t = %.2f'% t,
        sys.stdout.flush()
        global y
        y = ftcs2_step(y)
        line.set_ydata(y)
        return line

    ani = ani.FuncAnimation(fig, update, t_coords, interval=args.interval , repeat=False)
    plt.show()
elif args.solution:
    fig, ax = plt.subplots()
    ax.set_xlim([args.xmin, args.xmax])
    ax.set_ylim([args.ymin, args.ymax])
    y = exact_solution(0, x_coords, a, b, c)
    line, = ax.plot(x_coords,y)
    def update(t):
        print '\r @t = %.2f'% t,
        sys.stdout.flush()
        line.set_ydata(exact_solution(t, x_coords, a, b, c))
        return line
        
    ani = ani.FuncAnimation(fig, update, t_coords, interval=args.interval, repeat=False)
    plt.show()
else : #error view
    err = get_error()
    if args.show :
        plt.plot(t_coords[:len(err)], err)
        plt.show()



