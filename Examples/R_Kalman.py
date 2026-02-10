# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:33:51 2023

@author: y.ivashkevich
"""

import numpy as np
from scipy.linalg import block_diag

import Filters.MyKalman as kal
import Ballistic.Ballistic_Simulator as BS
from Ballistic.Ballistic_Simulator import PI

import param as p

import matplotlib.pyplot as plt
#%matplotlib inline
%matplotlib auto

def Kalman_1(dt, data, dtat_t):

    R_std = 30
    Q_std = 0.04 * 10**-3

    P = np.eye(9) * 5000.
    print("P =", "\n", P)

    F = np.array([[1, dt, dt**2/2,  0,  0,  0,       0,  0,  0],
                  [0,  1, dt,       0,  0,  0,       0,  0,  0],
                  [0,  0,  1,       0,  0,  0,       0,  0,  0],
                  [0,  0,  0,       1, dt, dt**2/2,  0,  0,  0],
                  [0,  0,  0,       0,  1, dt,       0,  0,  0],
                  [0,  0,  0,       0,  0,  1,       0,  0,  0],
                  [0,  0,  0,       0,  0,  0,       1, dt, dt**2/2],
                  [0,  0,  0,       0,  0,  0,       0,  1, dt],
                  [0,  0,  0,       0,  0,  0,       0,  0,  1]])
    print("F =", "\n", F)
    B = np.array([[1, dt, dt**2/2,  0,  0,  0,       0,  0,  0],
                  [0,  1, dt,       0,  0,  0,       0,  0,  0],
                  [0,  0,  1,       0,  0,  0,       0,  0,  0],
                  [0,  0,  0,       1, dt, dt**2/2,  0,  0,  0],
                  [0,  0,  0,       0,  1, dt,       0,  0,  0],
                  [0,  0,  0,       0,  0,  1,       0,  0,  0],
                  [0,  0,  0,       0,  0,  0,       1, dt, dt**2/2],
                  [0,  0,  0,       0,  0,  0,       0,  1, dt],
                  [0,  0,  0,       0,  0,  0,       0,  0,  1]])
    u = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    q = p.Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
    Q = block_diag(q, q, q)
    print("Q =", "\n", Q)

    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0]])
    print("H =", "\n", H)
 
    
    R = np.eye(3) * R_std**2
    print("R =", "\n", R)          
        
    f = kal.Kalman(dim_x=9, dim_m=3, P=P, F=F, B=B, u=u, Q=Q, H=H, R=R, m=data, dt_arr=data_t)
    for i in range(len(data)):
        #control_input(f)

        f.predict()
        f.update()

    print("xr =", data[-1, 0]/1000, "yr =", data[-1, 2]/1000)
    print("x =", f.get_filter()[-1, 0]/1000, "y =", f.get_filter()[-1, 6]/1000)

    return f

def Kalman_2(dt, data, data_t):
    R_std = 30
    Q_std = 0.04 * 10**3


    P = np.eye(9) * 5000.
    print("P =", "\n", P)

    F = np.array([[1, dt, dt**2/2,  0,  0,  0,       0,  0,  0],
                  [0,  1, dt,       0,  0,  0,       0,  0,  0],
                  [0,  0,  1,       0,  0,  0,       0,  0,  0],
                  [0,  0,  0,       1, dt, dt**2/2,  0,  0,  0],
                  [0,  0,  0,       0,  1, dt,       0,  0,  0],
                  [0,  0,  0,       0,  0,  1,       0,  0,  0],
                  [0,  0,  0,       0,  0,  0,       1, dt, dt**2/2],
                  [0,  0,  0,       0,  0,  0,       0,  1, dt],
                  [0,  0,  0,       0,  0,  0,       0,  0,  1]])
    print("F =", "\n", F)
    B = np.array([[1, dt, dt**2/2,  0,  0,  0,       0,  0,  0],
                  [0,  1, dt,       0,  0,  0,       0,  0,  0],
                  [0,  0,  1,       0,  0,  0,       0,  0,  0],
                  [0,  0,  0,       1, dt, dt**2/2,  0,  0,  0],
                  [0,  0,  0,       0,  1, dt,       0,  0,  0],
                  [0,  0,  0,       0,  0,  1,       0,  0,  0],
                  [0,  0,  0,       0,  0,  0,       1, dt, dt**2/2],
                  [0,  0,  0,       0,  0,  0,       0,  1, dt],
                  [0,  0,  0,       0,  0,  0,       0,  0,  1]])
    u = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])



        
    q = p.Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
    Q = block_diag(q, q, q)
    print("Q =", "\n", Q)

    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0]])
    print("H =", "\n", H)
    R = np.eye(3) * R_std**2
    print("R =", "\n", R)

    f = kal.Kalman(dim_x=9, dim_m=3, P=P, F=F, B=B, u=u, Q=Q, H=H, R=R, m=data, dt_arr=data_t)
    for i in range(len(data)):
        #control_input(f)
        f.step()
        #if f.n < f.N:
            #if (f.x[0] - f.m[f.n, 0] < 500): f.n += 1
            #if (np.array([f.x[0], f.x[3], f.x[6]]) - f.m[f.n]).all() < 500: f.n += 1

    print("xr =", data[-1, 0]/1000, "yr =", data[-1, 2]/1000)
    print("x =", f.get_filter()[-1, 0]/1000, "y =", f.get_filter()[-1, 6]/1000)

    return f

dt = 0.05624

data   = BS.load("r_1m")
data_t = BS.load("r_1t")

#data_r = BS.normalize_data(data_r)

for i in range(len(data[:, 0]) - 1):
    if (np.sqrt((data[i, 0] - data[i + 1, 0])**2 + 
                (data[i, 1] - data[i + 1, 1])**2 +
                (data[i, 2] - data[i + 1, 2])**2)) > 2000: 
        #print(i, abs(data[i, 1] - data[i + 1, 1]))
        data[i + 1] = data[i]

BS.show_from_data(data, "Реальные измерения")

#R.show()
    
f = Kalman_2(dt, data, data_t)    

f.plot_3d(3)
f.plot_r(name='x')
f.plot_r(axis_h=1, name='y')
f.plot_r(axis_h=2, name='z')

f.plot_l()

f.plot_e()
