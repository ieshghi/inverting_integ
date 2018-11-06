from numpy import *
import matplotlib.pyplot as plt
from scipy.integrate import RK45
def int_weights(n,dx = 1,method = 0):
    if method == 0: #trapezoidal rule
        a = dx*ones(n)
        a[0] = dx/2
        a[-1] = dx/2
    elif method == 1: 
        a = dx*ones(n)*1./3
        a[1::2] = dx*4./3
        a[0::2] = dx*2./3
        a[0] = dx/3
        a[-1] = dx/3
    return a

def gen_mat(n,ts,l): #makes nxn matrix of discretised integral
    #ts: array of t values
    #l: (relaxation time)^-1
    mat = zeros((n,n))
    dt = ts[1]-ts[0]
    mat[0,0] = dt
    for i in range(1,n):
        a = int_weights(i+1,dt)
        for j in range(i+1):
            mat[i,j] = a[j]*exp(-l*abs(ts[i]-ts[j]))
    return mat

def findv(rhs,t_i,t_f,l):
    n = rhs.size
    ts = linspace(t_i,t_f,n)
    m = gen_mat(n,ts,l)
    v = linalg.solve(m,rhs)
    return v

def find_r(r_i,v,t_i,t_f):
    y0 = r_i;
    r = RK45(v,t_i,y0,t_f)
    return r


