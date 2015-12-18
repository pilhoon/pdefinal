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
parser.add_argument('--dxinv', type=int, default=10)
args = parser.parse_args()

def sigma(limit, func):
    z = 0
    for i in range(limit):
        z += func(i)
    return z

def exact_solution(t, x) :
    return 3./8 + sigma(sigma_to, lambda l: (pow(-1, l)/(pi * (2*l+1)) + 2/(pi**2 * (2*l + 1)**2 ))*cos(pi*(2*l+1)*x)*exp(-pi**2*(2*l+1)**2*t)) + \
           sigma(sigma_to, lambda m: cos(2*pi*(2*m + 1)*x) / (pi**2*(2*m+1)**2 ) * exp(-4* pi**2 *(2*m+1)**2*t))

def ftcs(v, mu):
    vt = np.append(np.append(v[1],v), v[-2])
    for i in range(1,len(vt)-1):
        vt[i] = (1-2*mu)*vt[i] + mu*(vt[i-1] + vt[i+1])
    return vt[1:-1]

def exact_sol_view() :
    for t in np.arange(10.)/10:
        print exact_solution(t, -1)

    xx = np.arange(2000.)/1000 - 1
    print xx

    plt.plot( np.vectorize(exact_solution)(0.5, xx))
    plt.show()



#
# initialize global variables.
#

# u_t = u_xx , -1<x<1, 0<t<0.5
#mu = 0.4 
mu = 5
dx = 1./args.dxinv # 10 20 40 80 , dx = h
#mu = 1/dx #if lmbda = 1
x_num = int(2./dx)+1
x_coords = np.linspace(-1, 1, x_num)
dt = mu*dx*dx # dt = k
lmda = dt/dx
print 'lamda = ', lmda
print 'mu = ', mu
t_arr = np.linspace(0, 0.5, 0.5/dt + 1)

sigma_to = 100 # used to calculate exact solution. ideally, inf.
#pdb.set_trace()
#initial condition
def inity():
    y = []
    global x_coords
    for x in x_coords:
        if x < -1./2 or x > 1./2 :
            y = np.append(y, 0)
        elif abs(x) == 1./2:
            y = np.append(y, 1./4)
        else:
            y = np.append(y, 1-abs(x))
    return y


y = inity()
line = None

def update_line(t): #ftcs
    print '\r%.2f %%' % (100.*t/0.5),
    sys.stdout.flush()
    global dt, mu, y, line
    #pdb.set_trace()
    y[0] = exact_solution(t, -1) # v[0] : position x=-1
    y[-1] = 0 # v[last_index] = {position x=1} 
    y = ftcs(y, mu) # u(t+dt, x)
    line.set_ydata(y)

    #error
    return line,

def show_ftcs_transformation():
    #
    # run loop
    #
    global x_coords, y, line, t_arr
    y = inity()
    #plt.figure(1)
    fig, ax = plt.subplots()
    plt.ylim((-0.1, 1.1))
    plt.xlim((-1.1, 1.1))
    line, = ax.plot(x_coords, y)

    result_ani = ani.FuncAnimation(fig, update_line, t_arr, interval=1, repeat=False)
    #result_ani.save('6-3-11-a.mp4')
    plt.show()

def get_ftcs_error():
    global x_coords, t_arr, mu
    err = []
    y = inity()
    for t in t_arr:
        errs = y - np.vectorize(exact_solution)(t, x_coords)
        err += [linalg.norm(errs)]
        y = ftcs(y, mu)
    return err
    
# Crank Nicolson
cnmat = None
cnline = None
def cn_iter(t):
    global y, cnmat, cnline
    y[0] = exact_solution(t, -1)
    y[-1] = 0
    y = np.dot(cnmat, y)
    cnline.set_ydata(y)
    return cnline

def get_cn_mat(mu):
    y = inity()

    # construct cn mat
    leny = len(y)
    mat = np.zeros((leny, leny)) #float64
    mat[0,0] = 1+mu
    mat[0,1] = -mu
    mat[-1,-1] = 1+mu
    mat[-1,-2] = -mu
    for row in range(1,leny-1):
        mat[row,row-1] = -1.*mu/2
        mat[row,row] = 1+mu
        mat[row,row+1] = -1.*mu/2

    # get inverse
    cnmat = linalg.inv(mat)
    assert np.allclose(np.dot(mat, cnmat), np.eye(leny)) # assertion
    return cnmat

def show_crank_nicolson():
    global mu, y, t_arr, cnmat, x_coords, cnline
    y = inity()

    # construct cn mat
    cnmat = get_cn_mat(mu)

    fig, ax = plt.subplots()
    plt.ylim((-0.1, 1.1))
    plt.xlim((-1.1, 1.1))
    cnline, = ax.plot(x_coords, y)
    
    result_ani = ani.FuncAnimation(fig, cn_iter, t_arr, interval=25, repeat=False)
    result_ani.save('6-3-11-c-dxinv40.mp4')
    plt.show()

#plt.plot(get_ftcs_error())
#plt.show()

def get_cn_error():
    global t_arr, x_coords, mu
    y = inity()
    cnmat = get_cn_mat(mu)
    err = []
    for t in t_arr[1:]: #start from t_1 not t_0(init state)
        y = np.dot(cnmat, y) 
        err_arr = y - np.vectorize(exact_solution)(t, x_coords)
        err += [linalg.norm(err_arr)]
    return err


show_crank_nicolson()

#print get_cn_error()
#plt.plot(get_cn_error())
#plt.show()
