#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:29:55 2023

@author: yaroslavivashkevich
"""

import numpy as np
from time import time
from scipy.linalg import block_diag
from math import atan2
import sympy

import Filters.Extended_Kalman as kal
import Ballistic.Ballistic_Simulator as BS
from Ballistic.Ballistic_Simulator import PI
import Graphics.Graphic_manager as gm

import param as p

import matplotlib.pyplot as plt
#%matplotlib inline
%matplotlib auto


radar_pos = np.array([20000, 0, 25000])



def h_1(v):
    #print('v = ', v)
    if   v.shape[0] == 9: 

        
        v2 = np.array([v[0], v[3], v[6]])
        #print('v2 = ', v2)
        v2 -= radar_pos
        #print('v2 = ', v2)
        x, y, z = v2[0], v2[2], v2[1]
        if (x==0 and y==0 and z==0):
            return (np.array([0, 0, 0]))
        #print('xyz = ', x, y, z)
        ro = np.sqrt(x**2 + y**2 + z**2)
        r = ro
        phi = np.arctan(y/x)
        teta = np.arccos(z / ro)




        while phi > 2*PI:  phi -= 2*PI
        while phi < 0:     phi += 2*PI
        while teta > 2*PI: teta -= 2*PI
        while teta < 0:    teta += 2*PI
        if teta > PI: teta = 2*PI-teta
        
        if x < 0 and y < 0: phi +=PI
        if x < 0 and y > 0: phi -=PI
        
        #print("x->r", np.array([r, phi, teta]))
 
        return np.array([r, phi, teta])
    
    elif v.shape[0] == 3: 
        r, phi, teta = v[0], v[1], v[2]
        if (r == 0):
            return (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))

        while phi > 2*PI: phi -= 2*PI
        while phi < 0:    phi += 2*PI
        #print('rpt = ', r, phi, teta)
        

        x = r * np.sin(teta) * np.cos(phi)
        y = r * np.sin(teta) * np.sin(phi)
        z = r * np.cos(teta)
        
        a = np.array([x, z, y])
        #print('a = ', a)
        a += radar_pos
        
        #print("r->x", np.array([a[0], 0, a[4], 0, a[2], 0]))

        return np.array([a[0], 0, 0, a[1], 0, 0, a[2], 0, 0])
    
    return


def h_1_Jacobian(v, radar_pos):
    v2 = np.array([v[0], v[3], v[6]])
    v2 -= radar_pos
    x, y, z = v2[0], v2[2], v2[1]
        
    ro = np.sqrt(x**2 + y**2 + z**2)
    ro2 = x**2 + y**2 + z**2
    xy = np.sqrt(x**2+y**2)
    xy2 = x**2+y**2
    # h1 = np.sqrt(x**2 + y**2 + z**2)
    h1_dx = x/ro
    h1_dy = y/ro
    h1_dz = z/ro
    # h2 = np.arctan(y/x)
    h2_dx = -y/xy2
    h2_dy = x/xy2
    h2_dz = 0
    # h3 = np.arccos(z / ro)
    h3_dx = (x*z)/(ro*ro2*np.sqrt(1-(z**2/ro2)))
    h3_dy = (y*z)/(ro*ro2*np.sqrt(1-(z**2/ro2)))
    h3_dz = -xy2/(np.sqrt(xy2/ro2)*ro*ro2)
    
    HJacobian = np.array([[h1_dx, 0, 0, h1_dz, 0, 0, h1_dy, 0, 0],
                          [h2_dx, 0, 0, h2_dz, 0, 0, h2_dy, 0, 0],
                          [h3_dx, 0, 0, h3_dz, 0, 0, h3_dy, 0, 0]])

    return HJacobian

def normalize_angle(x, limit):
    while x >= 0+limit:
        x -= 2*np.pi
    while x < -2*PI+limit:
        x += 2*np.pi
    return x


def residual_h(x, y):
    z = np.zeros(len(x))
    z = x - y
    
    z[1] = normalize_angle(z[1], PI)
    z[2] = normalize_angle(z[2], PI)
    #print('z =', z)
    return z

def Extended_1(data, dt, r_s, phi_s, teta_s, Q_c):
    R_std = np.array([r_s, phi_s, teta_s]) * 1
    Q_std = 0.04 * 10**Q_c


    P = np.eye(9) * 500.
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

    #print("dt =", dt, 'Q_std =', Q_std)

    q = p.Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
    Q = block_diag(q, q, q)
    #print("Q =", "\n", Q)


    R = np.eye(3) * R_std**2
    #print("R =", "\n", R)
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    #print("x0 =", "\n", x0, "\n")

    #print(data)
    T = time()
    f = kal.ExtendedKalman(dim_x=9, dim_m=3, P=P, F=F, B=B, Q=Q, h=h_1, HJacobian=h_1_Jacobian, R=R, m=data, dt=dt, residual_h=residual_h)
    for i in range(len(data)):
        f.predict()
        f.update(args_j=(radar_pos))
    T = time() - T

        
    #print("xr =", data_r[-1, 0]/1000, "yr =", data_r[-1, 2]/1000)
    #print("x =", f.get_filter()[-1, 0]/1000, "y =", f.get_filter()[-1, 6]/1000)

    return f, T


def set_b():
    name = "P:\Diplom\Track\\ballist\\ballist_"
    save_name = "Extended_b_1_"
    Q_c = 1    
    return name, save_name, Q_c

def set_mo():
    name = "P:\Diplom\Track\manev_only_1\manev_only_"
    save_name = "Extended_mo1_3_"
    Q_c = 3
    return name, save_name, Q_c


name, save_name, Q_c = set_b()


k = 1
dt = k/10

r_s=10
phi_s=0.001
teta_s=0.001

radar_pos = np.array([20000, 0, 25000])

r_average = np.zeros(256)
c_average = np.zeros(256)
P_average = np.zeros(256)
t         = np.zeros(256)

for i in range(256):

    data = BS.load(name + str(i+1))
    data_r = BS.load(name + str(i+1) + 'rr')

    print("trajectory " + str(i))
    f, T = Extended_1(data_r, dt, r_s, phi_s, teta_s, Q_c)   
    
    #c = f.posterior - data
    #c2 = np.sqrt(c[0]**2 + c[3]**2 + c[6]**2)
    
    c_average[i] = f.plot_compare_res(new_data=data, plot=False)
    r_average[i] = f.plot_r3(plot=False)
    P_average[i] = f.plot_p(plot=False)
    
    
    
    t[i] = T
    print("r =", r_average[i], "c =", c_average[i], "p =", P_average[i], "t =", T)
    #print(f.x)
    #print(f.posterior[f.N-1])
    #print(f.prior[f.N-1])    
    #print(f.K)
    #print(f.y)
    #print(f.r)



num = 1
gm.plot_smth(r_average, name='r', label="Extended", save_name=save_name+"r", xlabel="Номер траектории", ylabel="м")
gm.plot_smth(P_average, name='p', label="Extended", color="green", save_name=save_name+"P", xlabel="Номер траектории", ylabel="м")
gm.plot_smth(t, name='t', label="Extended", color="yellow", save_name=save_name+"t", xlabel="Номер траектории", ylabel="с")
gm.plot_smth(c_average, name='c', label="Extended", save_name=save_name+"c", xlabel="Номер траектории", ylabel="м")


