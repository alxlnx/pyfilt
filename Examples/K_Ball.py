#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:59:54 2022

@author: yaroslavivashkevich
"""

from Filters.MyKalman import Kalman
from numpy.random import randn


import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import block_diag
#from filterpy.common import Q_discrete_white_noise

def Q_discrete_white_noise(dt, var):
    Q = [[0.25*dt**4, 0.5*dt**3, 0.5*dt**2],
         [0.5*dt**3,  dt**2,     dt],
         [0.5*dt**2,  dt,        1]]
        
    return block_diag(*[Q]*1) * var

class PosSensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = [vel[0], vel[1]]
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        
    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1] + 9.7  / 2.
        
        self.vel[1] -= 9.7
        
        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]
    
    def read_clear(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1] + 9.7 / 2.
        
        self.vel[1] -= 9.7
        
        return [self.pos[0],
                self.pos[1]]

R_std = 0.35 * 5
Q_std = 0.04 * 1

N = 25
sensor = PosSensor((0, 0), (2, 150), noise_std=R_std)
data = np.array([sensor.read() for _ in range(N)])
print("data =", "\n", data)

sensor2 = PosSensor((0, 0), (2, 150), noise_std=R_std)
data2 = np.array([sensor2.read_clear() for _ in range(N)])

dt = 1
P = np.eye(4) * 500.
P2 = np.eye(6) * 500.

print("P =", "\n", P)
F = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])
F2 = np.array([[1, dt, 0.5*dt**2, 0,  0, 0],
              [0,  1, dt,  0, 0, 0],
              [0,  0, 1, 0, 0, 0],
              [0,  0, 0,  1, dt, 0.5*dt**2],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]])
print("F =", "\n", F)
q = Q_discrete_white_noise( dt=dt, var=Q_std**2)
#q = np.eye(2)
Q = block_diag(q, q)
q2 = Q_discrete_white_noise( dt=dt, var=Q_std**2)
#q2 = np.eye(3)
Q2 = block_diag(q2, q2)
print("Q =", "\n", Q)
B = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])
B2 = np.array([[1, dt, 0.5*dt**2, 0,  0, 0],
              [0,  1, dt,  0, 0, 0],
              [0,  0, 1, 0, 0, 0],
              [0,  0, 0,  1, dt, 0.5*dt**2],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]])
u = np.array([0, 0, 0, -9.7])
u2 = np.array([0, 0, 0, 0, -9.7, 0])

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

H2 = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]])
print("H =", "\n", H)
R = np.eye(2) * R_std**2
print("R =", "\n", R)
#x0 = np.array([0, 0, 0, 0])
#print("x0 =", "\n", x0, "\n")

    
    
f = Kalman(dim_x=6, dim_m=2, P=P2, F=F2, Q=Q2, H=H2, R=R, B=B2, u=u2, m=data)
print("x0=", f.x, "\n")
f.predict()
print("x2=", f.x, "\n")
f.update()
f.calculate_all()



filter_data = f.get_filter()
print("filter =", "\n", filter_data)
      
f.plot_all(0, 1, 0, 3, data2[:, 0], data2[:, 1], loc=0)
f.plot_r()