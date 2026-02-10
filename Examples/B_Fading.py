# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:07:37 2023

@author: y.ivashkevich
"""

import numpy as np
from scipy.linalg import block_diag
from time import time

import Filters.Fading as fad
import Ballistic.Ballistic_Simulator as BS
from Ballistic.Ballistic_Simulator import PI
import Graphics.Graphic_manager as gm

import param

import matplotlib.pyplot as plt
#%matplotlib inline
%matplotlib auto

def Kalman_F(data, dt, alpha, Q_c):

    R_std = 30
    Q_std = 0.04 * Q_c

    P = np.eye(9) * 5000.
    #print("P =", "\n", P)

    F = np.array([[1, dt, dt**2/2,  0,  0,  0,       0,  0,  0],
                  [0,  1, dt,       0,  0,  0,       0,  0,  0],
                  [0,  0,  1,       0,  0,  0,       0,  0,  0],
                  [0,  0,  0,       1, dt, dt**2/2,  0,  0,  0],
                  [0,  0,  0,       0,  1, dt,       0,  0,  0],
                  [0,  0,  0,       0,  0,  1,       0,  0,  0],
                  [0,  0,  0,       0,  0,  0,       1, dt, dt**2/2],
                  [0,  0,  0,       0,  0,  0,       0,  1, dt],
                  [0,  0,  0,       0,  0,  0,       0,  0,  1]])
    #print("F =", "\n", F)
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
    
    q = param.Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
    Q = block_diag(q, q, q)
    #print("Q =", "\n", Q)

    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0]])
    #print("H =", "\n", H)
 
    
    R = np.eye(3) * R_std**2
    #print("R =", "\n", R)  
    
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    #print("x0 =", "\n", x0, "\n")

        
    T = time()
    f = fad.Fading(dim_x=9, dim_m=3, P=P, F=F, B=B, u=u, Q=Q, H=H, R=R, m=data)
    for i in range(len(data)):
        
        f.predict(alpha)
        f.update()
    T = time() - T

    #print("xr =", data_r[-1, 0]/1000, "yr =", data_r[-1, 2]/1000)
    #print("x =", f.get_filter()[-1, 0]/1000, "y =", f.get_filter()[-1, 6]/1000)

    return f, T


def set_b():
    name = "P:\Diplom\Track\\ballist\\ballist_"
    save_name = "Fading_b_1_"
    Q_c = 1    
    return name, save_name, Q_c

def set_mo():
    name = "P:\Diplom\Track\manev_only_1\manev_only_"
    save_name = "Fading_mo1_3_"
    Q_c = 3
    return name, save_name, Q_c


name, save_name, Q_c = set_b()

k = 1
dt = k/10
radar_pos = np.array((20000, 0, 25000))


r_average = np.zeros(256)
c_average = np.zeros(256)
P_average = np.zeros(256)
t         = np.zeros(256)

#alpha=0.999
#alpha=1
alpha=1.001

for i in range(256):
    data = BS.load(name + str(i+1))
    data_r = BS.load(name + str(i+1) + 'r')

    print("trajectory " + str(i))
    f, T = Kalman_F(data_r, dt, alpha, Q_c)    

    c_average[i] = f.plot_compare_res(new_data=data, plot=False)
    r_average[i] = f.plot_r3(plot=False)
    P_average[i] = f.plot_p(plot=False)
    t[i] = T
    print("r =", r_average[i], "c =", c_average[i], "p =", P_average[i], "t =", T)

gm.plot_smth(r_average, name='r', label="Fading", save_name=save_name+"r", xlabel="Номер траектории", ylabel="м")
gm.plot_smth(P_average, name='p', label="Fading", color="green", save_name=save_name+"P", xlabel="Номер траектории", ylabel="м")
gm.plot_smth(t, name='t', label="Fading", color="yellow", save_name=save_name+"t", xlabel="Номер траектории", ylabel="с")
gm.plot_smth(c_average, name='c', label="Fading", save_name=save_name+"c", xlabel="Номер траектории", ylabel="м")
#f = Kalman(dt, data_r)    

    
#f  = Kalman_F(dt, alpha=0.99)    
#f2 = Kalman_3d_1(dt, alpha=1.00)    
#f3 = Kalman_3d_1(dt, alpha=1.01)    




