# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:36:34 2023

@author: y.ivashkevich
"""

import numpy as np
from time import time
from scipy.linalg import block_diag

import Filters.MyKalman as kal
import Ballistic.Ballistic_Simulator as BS
from Ballistic.Ballistic_Simulator import PI
import Graphics.Graphic_manager as gm

import param as p

import matplotlib.pyplot as plt
#%matplotlib inline
%matplotlib auto


def Kalman(dt, data, Q_c):
    R_std = 30
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
    #x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    #print("x0 =", "\n", x0, "\n")

        
    T = time()
    f = kal.Kalman(dim_x=9, dim_m=3, P=P, F=F, B=B, u=u, Q=Q, H=H, R=R, m=data, dt=dt)
    for i in range(len(data)):
        #control_input(f)

        f.predict()
        f.update()
    T = time() - T


    #print("xr =", data_r[-1, 0]/1000, "yr =", data_r[-1, 2]/1000)
    #print("x =", f.get_filter()[-1, 0]/1000, "y =", f.get_filter()[-1, 6]/1000)

    return f, T

def set_b():
    #name = "/Users/yaroslavivashkevich/Code/Diplom/Track/ballist/ballist_"
    name = "P:\Diplom\Track\\ballist\\ballist_"
    save_name = "Kalman_b_1_"
    Q_c = 1    
    return name, save_name, Q_c

def set_mo():
    name = "P:\Diplom\Track\manev_only_1\manev_only_"
    save_name = "Kalman_mo1_3_"
    Q_c = 3
    return name, save_name, Q_c


name, save_name, Q_c = set_b()


radar_pos = np.array((20000, 0, 25000))




k = 1
dt = k/10


r_average = np.zeros(256)
c_average = np.zeros(256)
P_average = np.zeros(256)
t         = np.zeros(256)

for i in range(256):
    data = BS.load(name + str(i+1))
    data_r = BS.load(name + str(i+1) + 'r')

    print("trajectory " + str(i))
    f, T = Kalman(dt, data_r, Q_c)    

    c_average[i] = f.plot_compare_res(new_data=data, plot=False)
    #r_average[i] = f.plot_r3(plot=False)
    P_average[i] = f.plot_p(plot=False)
    t[i] = T
    print("r =", r_average[i], "c =", c_average[i], "p =", P_average[i], "t =", T)
    f.plot_3d(f.dim_x, radar_pos=radar_pos)

gm.plot_smth(r_average, name='r', label="Kalman", save_name=save_name+"r", xlabel="Номер траектории", ylabel="м")
gm.plot_smth(P_average, name='p', label="Kalman", color="green", save_name=save_name+"P", xlabel="Номер траектории", ylabel="м")
gm.plot_smth(t, name='t', label="Kalman", color="yellow", save_name=save_name+"t", xlabel="Номер траектории", ylabel="с")
gm.plot_smth(c_average, name='c', label="Kalman", save_name=save_name+"c", xlabel="Номер траектории", ylabel="м")
#f = Kalman(dt, data_r)    



'''
for i in range(256):
    data = BS.load("/Users/yaroslavivashkevich/Code/Diplom/Track/manev_2/manev2_" + str(i+1))
    data_r = BS.load("/Users/yaroslavivashkevich/Code/Diplom/Track/manev_2/manev2_" + str(i+1) + 'r')
    data_rr = BS.load("/Users/yaroslavivashkevich/Code/Diplom/Track/manev_2/manev2_" + str(i+1) + 'rr')

    f, T = Kalman(dt, data)    

    for ii in range(len(data)):
        if f.e_arr[ii] > 5 and ii > 200:
            cut = ii - 50
            print(i, cut)
            data = np.array(data[cut:-1])
            data_r = np.array(data_r[cut:-1])
            data_rr = np.array(data_rr[cut:-1])
            name = "/Users/yaroslavivashkevich/Code/Diplom/Track/manev_only_2/manev2_only_" + str(i+1)
            BS.save(data, name)
            BS.save(data_r, name +'r')
            BS.save(data_rr, name + 'rr')
            break
'''     

f.plot_e()  
#BS.show_from_data(data)
        
    


#R = BS.Radar(radar_pos, k)
#R.run(data_r, r_s=10, phi_s=0.001, teta_s=0.001)
# берем данные, переведенные в МЗСК, тк мы тут не можем работать с нелинейными преобразованиями
#R_data = R.get_data()
#data = R.get_data()        

#f = Kalman(dt)    


