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
parser.add_argument('--type', type=int, default=2) 
parser.add_argument('--simulate', action='store_true')  #if not show error
parser.add_argument('--interval', type=int , default=25)  #if not show error
parser.add_argument('--show', action='store_true', help='show error graph')
parser.add_argument('--tmax', type=int, default=1)
parser.add_argument('--solution', action='store_true')
parser.add_argument('--saveto', type=str)
parser.add_argument('--framelimit', type=int)
args = parser.parse_args()

dx = 1./args.h # x: -1 ~ 1
x_coords = np.linspace(-1, 1, args.h*2 + 1)
dt = 1./args.k
t_coords = np.linspace(0, args.tmax, args.k*args.tmax + 1)
if args.framelimit:
    t_coords = t_coords[0:args.framelimit]
a = args.a
b = args.b
c = args.c
assert a>0 and c>0

def exact_solution(t, x, a, b, c): #x : array
    k = np.multiply(1.*c/(2*b), np.subtract(x, a*t))
    return np.subtract(a, np.multiply(c, np.tanh(k)))

def yinit():
    return exact_solution(0, x_coords, a, b, c)
# mu = dt / (dx*dx)
# lmbda = dt / dx

print '---------------------'
print ' a = ', a
print ' b = ', b
print ' c = ', c
print ' c/b = ', c/b
print ' a/b = ', a/b
print 
print ' t = 0 ~', args.tmax, '(len =', len(t_coords), ')'
print ' x = -1 ~ 1 (len =', len(x_coords), ')'
print ' dx = ', dx, '(h =', args.h, ')'
print ' dt = ', dt, '(k =', args.k, ')'
print 
print ' dx*c/2b = ', (dx*c)/(2*b)
print ' dx*a/2b = ', (dx*a)/(2*b)
print 
critv = 1. - 2.*b*dt/(dx**2)
if critv < 0 or critv >= 1:
    print '\033[31m',
print ' 1- 2b*dt/dx**2 = ', critv,
print '\033[0m'
derivatives = np.multiply(-0.5*(c**2)/b, np.subtract(1, np.power(np.tanh(np.multiply(1.*c/(2*b),x_coords)), 2)))
#print ' du/dx = ', derivatives
print ' real max slope =', max(abs(derivatives))
y = yinit()
y1 = np.append(y[1:], 0)
print ' approx max slope =', max(abs(y-y1)[:-1]/dx)
print ' cf slope = c/4b =', 0.25*c/b

def ftcs1_step(y): # 1/2 * (u^2)_x = u*u_x # very bad
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
        if max(err) > (a+c)*10:
            print ' \033[31mdiverge at ', idx , '/', len(t_coords)-1, '\033[0m'
            break;
    print ' error avg =', np.average(err)
    print '       std =', np.std(err)
    print '       max =', max(err)
    print '       min =', min(err)
    print ' final mean =', np.average(y) , '(cf. a+c=', a+c,')'
    return err

#
# exact solution view
#

if args.simulate :
    #
    # ftcs view
    #
    print ' interval = ', args.interval , ' ms'
    fig, ax = plt.subplots()
    ax.set_xlim([-1, 1])
    ax.set_ylim([1.1*(a-c), 1.1*(a+c)])
    y = exact_solution(0, x_coords, a, b, c)
    line, = ax.plot(x_coords,y)
    ##plt.plot(x_coords, exact_solution(0.5, x_coords, 1,1,1))
    def update(t):
        print '\r @t = %.2f'% t,
        sys.stdout.flush()
        global y
        if args.type == 1:
            y = ftcs1_step(y)
        else: #default. more stable
            y = ftcs2_step(y)
        line.set_ydata(y)
        return line

    line_ani = ani.FuncAnimation(fig, update, t_coords, interval=args.interval , repeat=False)
    if args.saveto :
        line_ani.save(args.saveto)
    else:
        plt.show()
elif args.solution:
    fig, ax = plt.subplots()
    ax.set_xlim([-1, 1])
    ax.set_ylim([1.1*(a-c), 1.1*(a+c)])
    y = exact_solution(0, x_coords, a, b, c)
    line, = ax.plot(x_coords,y)
    def update(t):
        print '\r @t = %.2f'% t,
        sys.stdout.flush()
        line.set_ydata(exact_solution(t, x_coords, a, b, c))
        return line
        
    line_ani = ani.FuncAnimation(fig, update, t_coords, interval=args.interval, repeat=False)
    if args.saveto :
        line_ani.save(args.saveto)
    else:
        plt.show()
else : #error view
    err = get_error()
    if args.show :
        plt.plot(t_coords[:len(err)], err)
        plt.show()



