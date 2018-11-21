from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np

def int_weights(n,dx = 1,method = 0):
    if method == 0: #trapezoidal rule
        a = dx*ones(n)
        a[0] = dx/2
        a[-1] = dx/2
    elif method == 1: 
        a = dx*ones(n)
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
        mat[i,0:(i+1)] = a[:(i+1)]*exp(-l*abs(ts[i]-ts[:(i+1)]))
    return mat*n

def findv(rhs,t_i,t_f,l):
    n = rhs.size
    ts = linspace(t_i,t_f,n)
    m = gen_mat(n,ts,l)
    mi = abs(linalg.inv(m))
    v = mi.dot(rhs)

    return v

def find_r(r_i,v,t_i,t_f):
    dt = (t_f-t_i)/v.size
    r = zeros(v.shape)
    r[0] = r_i
    for i in (range(1,v.size)):
        r[i] = r[i-1] + v[i-1]*dt
    return r

def corrnoise(n,l):
    c = exp(-l)
    nums_out = zeros(n)
    g = randn(n)
    for i in range(1,n):
        nums_out[i] = nums_out[i-1]*c + sqrt(1-c**2)*g[i]

    return nums_out

def simulate_maxwell(n,l):
    t0 = 0.
    t1 = 1.
    noise_x = corrnoise(n,l)
    noise_y = corrnoise(n,l)
    vx = findv(noise_x,t0,t1,l)
    vy = findv(noise_y,t0,t1,l)
    rx = find_r(0.,vx,t0,t1)
    ry = find_r(0.,vy,t0,t1)
    return rx,ry

def msd(trajx,trajy):
    traj = vstack((trajx,trajy))
    n = trajx.size
    shifts = linspace(1,n-1,n-1)
    msds = np.zeros(shifts.size)
    for i, shift in enumerate(shifts):
        shift = int(shift)
        diffs = traj - roll(traj,-shift)
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
    return msds
