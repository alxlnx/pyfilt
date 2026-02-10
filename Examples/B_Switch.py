# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:45:14 2023

@author: y.ivashkevich
"""


import numpy as np
from scipy.linalg import block_diag
from time import time

import Filters.MyKalman as kal
import Filters.Sigma_Kalman as sig
import Graphics.Graphic_manager as gm


import Filters.Switch as sw
import Ballistic.Ballistic_Simulator as BS
from Ballistic.Ballistic_Simulator import PI

import param as p

import matplotlib.pyplot as plt
#%matplotlib inline
%matplotlib auto











def f_2(x, dt):
    F = np.array([[1, dt, dt**2/2,  0,  0,  0,       0,  0,  0],
                  [0,  1, dt,       0,  0,  0,       0,  0,  0],
                  [0,  0,  1,       0,  0,  0,       0,  0,  0],
                  [0,  0,  0,       1, dt, dt**2/2,  0,  0,  0],
                  [0,  0,  0,       0,  1, dt,       0,  0,  0],
                  [0,  0,  0,       0,  0,  1,       0,  0,  0],
                  [0,  0,  0,       0,  0,  0,       1, dt, dt**2/2],
                  [0,  0,  0,       0,  0,  0,       0,  1, dt],
                  [0,  0,  0,       0,  0,  0,       0,  0,  1]])
    u = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    
    return F @ x + F @ u

def h_2(v):
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

def Kalman_1(data, dt):

    R_std = 30
    Q_std = 0.04 * 10**1

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
    
    q = p.Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
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

    f = kal.Kalman(dim_x=9, dim_m=3, P=P, F=F, B=B, u=u, Q=Q, H=H, R=R, m=data)

    return f

def Kalman_2(data, dt):
    R_std = 30
    Q_std = 0.04 * 10**3


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
        
    q = p.Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
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
        
    f = kal.Kalman(dim_x=9, dim_m=3, P=P, F=F, B=B, u=u, Q=Q, H=H, R=R, m=data)

    return f


def Sigma_1(data, dt, r_s, phi_s, teta_s):
    R_std = np.array([r_s, phi_s, teta_s]) * 1
    Q_std = 0.04 * 10**1

    P = np.eye(9) * 500.
    #print("P =", "\n", P)


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


    #print('dt =', dt, 'Q_std =', Q_std)

    q = p.Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
    #print('q', q)
    Q = block_diag(q, q, q)
    #print("Q =", "\n", Q)

    R = np.eye(3) * R_std**2
    #print("R =", "\n", R)
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    #print("x0 =", "\n", x0, "\n")
    #print('data', data)
        
        
    f = sig.SigmaKalman(dim_x=9, dim_m=3, P=P, f=f_2, B=B, Q=Q, h=h_2, R=R, m=data, dt=dt)

    return f

def Sigma_2(data, dt, r_s, phi_s, teta_s):
    R_std = np.array([r_s, phi_s, teta_s]) * 1
    Q_std = 0.04 * 10**3

    P = np.eye(9) * 500.
    #print("P =", "\n", P)


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


    #print('dt =', dt, 'Q_std =', Q_std)

    q = p.Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
    #print('q', q)
    Q = block_diag(q, q, q)
    #print("Q =", "\n", Q)

    R = np.eye(3) * R_std**2
    #print("R =", "\n", R)
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    #print("x0 =", "\n", x0, "\n")
    #print('data', data)
        
        
    f = sig.SigmaKalman(dim_x=9, dim_m=3, P=P, f=f_2, B=B, Q=Q, h=h_2, R=R, m=data, dt=dt)

    return f

r_s=10
phi_s=0.001
teta_s=0.001

def run_K(data):
    f1 = Kalman_1(data, dt)    
    f2 = Kalman_2(data, dt)

    filters = [f1, f2]  
    return filters
    
def run_S(data):
    f1 = Sigma_1(data, dt, r_s, phi_s, teta_s)    
    f2 = Sigma_2(data, dt, r_s, phi_s, teta_s)

    filters = [f1, f2]  
    return filters


def set_mK():
    name = "P:\Diplom\Track\manev_2\manev2_"
    save_name = "Switch_K_"
      
    return name, save_name, 1

def set_mS():
    name = "P:\Diplom\Track\manev_2\manev2_"
    save_name = "Switch_S_"
      
    return name, save_name, 2


name, save_name, r = set_mK()


k = 1
dt = k/10

radar_pos = np.array((20000, 0, 25000))

#f1 = Kalman_1(dt)    
#f2 = Kalman_2(dt)

#f3 = Sigma_1(data_r, r_s=10, phi_s=0.001, teta_s=0.001, dt=dt)    
#f4 = Sigma_2(data_r, r_s=10, phi_s=0.001, teta_s=0.001, dt=dt)

r_average = np.zeros(256)
c_average = np.zeros(256)
P_average = np.zeros(256)
t         = np.zeros(256)

for i in range(256):
    data = BS.load(name + str(i+1))
    if r == 1:  data_r = BS.load(name + str(i+1) + 'r')
    if r == 2:  data_r = BS.load(name + str(i+1) + 'rr')
    print("trajectory " + str(i))
    
    T = time()
    filters = run_K(data_r)  
    af = sw.Switch(filters, eps_max=10, dt=dt)
    af.run()
    T = time() - T

    c_average[i] = af.plot_compare_res(new_data=data, plot=False)
    #r_average[i] = af.plot_r3(plot=False)
    #P_average[i] = af.plot_p(plot=False)
    t[i] = T
    print("c =", c_average[i], "t =", T)

#gm.plot_smth(r_average, name='r', label="Switch", save_name="Switch_k_K_r", xlabel="Номер траектории", ylabel="м")
#gm.plot_smth(P_average, name='p', label="Switch", color="green", save_name="KSwitch_b_K_P", xlabel="Номер траектории", ylabel="м")
gm.plot_smth(t, name='t', label="Switch", color="yellow", save_name=save_name+"t", xlabel="Номер траектории", ylabel="с")
gm.plot_smth(c_average, name='c', label="Switch", save_name=save_name+"c", xlabel="Номер траектории", ylabel="м")



    
