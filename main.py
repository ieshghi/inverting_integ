from numpy import *
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from numpy.random import randn
import numpy as np

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
        mat[i,0:(i+1)] = a[:(i+1)]*exp(-l*abs(ts[i]-ts[:(i+1)]))
    return mat*n

def findv(rhs,t_i,t_f,l):
    n = rhs.size
    ts = linspace(t_i,t_f,n)
    m = gen_mat(n,ts,l)
    v = linalg.solve(m,rhs)
    signswitch = ones(v.size)
    signswitch[1::2] = -1

    return v*signswitch

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

def simulate_maxwell(n):
    t0 = 0
    t1 = 1
    l = 2.
    noise_x = corrnoise(n,l)
    noise_y = corrnoise(n,l)
    vx = findv(noise_x,t0,t1,l)
    vy = findv(noise_y,t0,t1,l)
    rx = find_r(0,vx,0,1)
    ry = find_r(0,vy,0,1)
    return rx,ry

def msd(rx,ry):
    r = sqrt(rx**2+ry**2)
    diffr = diff(r) #this calculates r(t + dt) - r(t)
    diff_sq = diffr**2
    MSD = mean(diff_sq)
    return MSD

def autocorrFFT(x):
  N=len(x)
  F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
  PSD = F * F.conjugate()
  res = np.fft.ifft(PSD)
  res= (res[:N]).real   #now we have the autocorrelation in convention B
  n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
  return res/n #this is the autocorrelation in convention A

def msd_fft(r):
  N=len(r)
  D=np.square(r).sum(axis=1)
  D=np.append(D,0)
  S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
  Q=2*D.sum()
  S1=np.zeros(N)
  for m in range(N):
      Q=Q-D[m-1]-D[N-m]
      S1[m]=Q/(N-m)
  return S1-2*S2

