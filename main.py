from numpy import *
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np
from scipy.special import gamma as spgam

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
    mi = abs(linalg.inv(m)) #Why absolute value?
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

def simulate_maxwell(n,l,t0,t1):
    l0 = l*abs((t0-t1)/n)
    noise_x = corrnoise(n,l0)
    noise_y = corrnoise(n,l0)
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

def maxwell_msd(n,l,nav = 1):
    ms = zeros((n-1,nav))
    for i in range(nav):
        n = int(n)
        t0 = .25
        t1 = .25*60
        rx,ry = simulate_maxwell(n,l,t0,t1)
        m = msd(rx,ry)
        ms[:,i] = m

    t = linspace(t0,t1,n-1)
    return t,mean(ms,1)

def logderive(x,f,width): #stolen from Kilfoil lab code, translated to Python
    np = x.size
    df = zeros(np)
    ddf = zeros(np)
    f2 = zeros(np)
    lx = log(x)
    ly = log(f)
    for i in range(np):
        we = exp(-(lx-lx[i])**2/(2*width**2)) 
        ww = [we>.03]
        res = polyfit(lx[ww],ly[ww],2,w = we[ww])
        f2[i] = exp(res[2] + res[1]*lx[i] + res[0]*(lx[i]**2))
        df[i] = res[1]+(2*res[0]*lx[i])
        ddf[i] = 2*res[0]

    return f2,df,ddf

def calc_moduli(tau,msd,a,T):
    clip = .03
    width = .7
    k = 1.38065e-23
    am = a*1e-6
    dt = tau
    omega = 1./dt
    msdm = msd*1e-12
    C = k*T/(pi*am) #assume 3d
    foo = (np.pi/2)-1
    m,d,dd = logderive(dt,msdm,width)
    Gs = C/((m*spgam(1+d))*(1 +(dd/2)))
    Gs[Gs<0] = 0
    g,da,dda = logderive(omega,Gs,width);
    Gp  = g*(1./(1+dda))*(cos((pi/2)*da)-foo*da*dda)
    Gpp = g*(1./(1+dda))*(sin((pi/2)*da)-foo*(1-da)*dda)
    
    w = [Gp < Gs*clip]
    nw = len(w)
    if nw > 0:
        Gp[w]=0
    w = [Gpp < Gs*clip]
    nw = len(w)
    if nw > 0:
        Gpp[w]=0

    return Gp,Gpp 


def genmod(n,l,nav):
    t,msd = maxwell_msd(n,l,nav)
    gp,gpp = calc_moduli(t,msd,1.,300.)
    w = t**(-1)
    return w,gp,gpp
