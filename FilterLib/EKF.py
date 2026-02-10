#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 03:37:16 2023

@author: yaroslavivashkevich
"""

import numpy as np
from FilterLib.Kalman import Kalman

class EKF(Kalman):
    '''
    Расширенный фильтр Калмана (но линейный в predict()!)
    Attributes:
        dim_x (int) : количество переменных состояния 
        dim_m (int) : количество переменных измерения
        
        P (np.array(float, ndmin=2)) : матрица ковариации текущего состояния
        F (np.array(float, ndmin=2)) : функция перехода состояния (Xn = F*X)
        Q (np.array(float, ndmin=2)) : ковариационная матрица шума процесса (переходная ковариация)
        B (np.array(float, ndmin=2)) : control function (dx = B*u)
        u (np.array(float, ndmin=1)) : control input
        H (np.array(float, ndmin=2)) : матрица измерения (y = z - H*x, y - невязка, z - измерение, x - априорное состояние)                      
        h (function)                 : функция измерения
        R : np.array(float, ndmin=2) - матрица коварации для шума измерений                                  
        S : np.array(float, ndmin=2) - system uncertainty (общий шум)
        K : np.array(float, ndmin=2) - kalman gain, содержит значения от 0 до 1        
        y : np.array(float, ndmin=1) - невязка
    
        x : np.array(float, ndmin=1) - текущем состоянии, dim=(x, 1), при инициализации - предсказание начального положения
        z : np.array(float, ndmin=1) - данные о текущем измерении, dim=(x, 1)
        m : np.array(float, ndmin=2) - значения измерений,         dim=(m, 1)
        
        prior       : np.array(float, ndmin=2) - значения предсказаний,  dim=(x, 1)
        posterior   : np.array(float, ndmin=2) - значения фильтра, dim=(x, 1)
        P_prior     : np.array(float, ndmin=3) - значения предсказанной ковариационной матрицы,   dim=(x, x)
        P_posterior : np.array(float, ndmin=3) - значения отфильтрованной ковариационной матрицы, dim=(x, x)

        res_arr     : np.array(float, ndmin=2) - массив невязок
        l_arr       : np.array(float, ndmin=1) - массив функций правдоподобия
        e_arr       : np.array(float, ndmin=1) - массив эпсилон (невязка в квадрате, деленная на общий шум)
        r           : np.array(float, ndmin=1) - невязка в декартоввых координатах

        n           : int - шаг
        N           : int - шагов всего
        dt          : float - шаг времени
        dt_arr      : np.array(float, ndmin=1) - массив времени, если None, то всегда используется dt
        
        HJacobian   : function - якобиан функции измерения h
        residual_h  : function - функции разности векторов

        predict_stage : bool - защита от пвоторного использования update() и predict()
        multi_update  : bool - игнорирует защиту от повторого использования

    -----------------------    
    Methods:
        add_measurement   - добавление измернеия
        predict           - этап предсказания
        update            - этап обновления
        step              - шаг фильтра, предсказание + обновление
        calculate_all     - запускает фильтр до конца массива измерений
        likelihood        - функция правдоподобия
        epsilon           - невязка в квадрате, деленная на общий шум
        
        _first_step       - первый шаг фильтра
        _count_parameters - вычисление и сохранение различных парамаетров
        
        save_graphic      - сохранение графика задынных данных
        P_medium          - Среднее значение главной диагонали матрицы P
        plot_p            - построение и сохранение графика средних значений диагонали матрицы ковариации
        plot_r            - построение и сохранение графика невязки
        plot_r3           - построение и сохранение графика абсолютной невязки по всем координатам
        plot_l            - построение и сохранение графика значений функции правдоподобия
        plot_e            - построение и сохранение графика эпсилон
        plot_3d           - построение и сохранение отфильтрованных значений в 3d
        plot_compare_res  - построение и сохранение разницы между отфильтрованными и заданными значениями
    '''

    def __init__(self, dim_x : int, dim_m : int, 
                 HJacobian, h,
                 P=None, F=None, Q=None, B=None, 
                 u=None, R=None, x0=None, m=None,
                 residual_h=None, dt=None, dt_arr=None):
        
        # dim_x - количество переменных состояния 
        self.dim_x : int = dim_x
        # dim_m - количество переменных измерения
        self.dim_m : int = dim_m
        
        # P - матрица ковариации текущего состояния
        self.P : np.array(float, ndmin=2) = np.eye(dim_x)
        # F - функция перехода состояния (Xn = F*X)
        self.F : np.array(float, ndmin=2) = np.eye(dim_x)       
        # Q - ковариационная матрица шума процесса (переходная ковариация)
        self.Q : np.array(float, ndmin=2) = np.eye(dim_x)
        # B - control function (dx = B*u)
        self.B : np.array(float, ndmin=2) = np.zeros((dim_x, dim_x))
        # u - control input
        self.u : np.array(float, ndmin=1) = np.zeros(dim_x)
        # H - матрица измерения (y = z - H*x, y - невязка, z - измерение, x - априорное состояние)     
        self.H : np.array(float, ndmin=2) = np.zeros((dim_m, dim_x))
        # h - функция измерения                 
        self.h = None # function
        # R - матрица коварации для шума измерений                                  
        self.R : np.array(float, ndmin=2) = np.eye(dim_m)
        # S - system uncertainty 
        self.S : np.array(float, ndmin=2) = np.zeros((dim_m, dim_m)) 
        # K - kalman gain, содержит значения от 0 до 1
        self.K : np.array(float, ndmin=2) = np.zeros((dim_x, dim_m)) 
        # y - невязка
        self.y : np.array(float, ndmin=1) = np.zeros(dim_m) 
                 
        # P_prior - значения предсказанной ковариационной матрицы,       dim=(x, x)
        self.P_prior     : np.array(float, ndmin=2) = np.zeros((dim_x, dim_x))
        # P_posterior - значения отфильтрованной ковариационной матрицы, dim=(x, x)
        self.P_posterior : np.array(float, ndmin=2) = np.zeros((dim_x, dim_x))
        
        # x - данные о текущем состоянии, dim=(x, 1)
        # при инициализации - предсказание начального положения
        self.x           : np.array(float, ndmin=1) = None
        # z - данные о текущем измерении, dim=(x, 1)
        self.z           : np.array(float, ndmin=1) = None
        # m - значения измерений,         dim=(m, 1)
        self.m           : np.array(float, ndmin=2) = None                      
        # prior - значения предсказаний,  dim=(x, 1)
        self.prior       : np.array(float, ndmin=2) = None
        # posterior - значения фильтра,   dim=(x, 1)
        self.posterior   : np.array(float, ndmin=2) = None
    
        # массив невязок
        self.res_arr     : np.array(float, ndmin=2) = None
        # массив функций правдоподобия
        self.l_arr       : np.array(float, ndmin=1) = None  
        # массив эпсилон (невязка в квадрате, деленная на общий шум)
        self.e_arr       : np.array(float, ndmin=1) = None    
        # неввязка в декартоввых координатах
        self.r           : np.array(float, ndmin=1) = None 
 
        # n - шаг
        self.n  : int = 0
        # N - шагов всего
        self.N  : int = 0
        # dt - шаг времени
        self.dt : float = 1
        # dt_arr - массив времени
        self.dt_arr = None
        
        # якобиан
        self.HJacobian = None # function
        
        # функции разности векторов
        self.residual_h = np.subtract # function
        
        #защита от пвоторного использования update() и predict()
        self.predict_stage : bool = True
        #игнорирует защиту от повторого использования
        self.multi_update  : bool = False

        if P  is not None: self.P = P
        if F  is not None: self.F = F
        if Q  is not None: self.Q = Q
        if B  is not None: self.B = B
        if u  is not None: self.u = u
        if h  is not None: self.h = h  
        else :
            print("Error! No h!")
            return 
        if R  is not None: self.R = R
        if x0 is not None: self.x = x0
        if dt is not None: self.dt= dt
        if dt_arr is not None: self.dt = dt_arr[0]
        if m  is not None: 
            self.m = m           
            self.N = len(m)
        if HJacobian is not None: self.HJacobian = HJacobian
        else: 
            print("Error! No HJacobian!")
            return
        if residual_h is not None: self.residual_h=residual_h
        
        return
    


        
    def predict(self):    
        '''
        Этап предсказания
        
        X_prior = F*X_post + B*u
        P_prior = F*P_post*F^T + Q

        Если это первый шаг фильтра, также вызывает функцию _first_step

        Returns
        -------
        None.

        '''                 
        if self.predict_stage == False:
            print("Error! Double call of predict!")
            return
        if self.n >= self.N:
            print("Error! No more data!")
            return
        
        if self.n == 0:
            self._first_step()

        # X_prior = F*X_post + B*u
        self.x = self.F @ self.x + self.B @ self.u
        
        # P_prior = F*P_post*F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.prior   = np.append(self.prior, [self.x], axis=0)
        #print(self.P_prior.shape, self.P.shape)
        #self.P_prior = np.append(self.P_prior, [self.P], axis=0)
        self.P_prior = np.append(self.P_prior, self.P, axis=0)
        
        
        self.predict_stage = False
        return
        
    def update(self, args_j=(), args_h=()):
        '''
        Этап обновления    
        H = HJacobian(x) = dh/dx(x)   
        S = H*P_prior*H^T + R     
        K = P_prior*H^T*S^(-1)    
        y = m - h(X_prior)    
        X_post = X_prior + K * y     
        P_post = (I - K*H) * P_prior     
        
        Также вычисляются и сохраняются такие параметры, как невязка и функция правдоподобия

        Attributes:
            args_j (arguments, optional) :
                аргументы функции HJacobian.
            args_h (arguments, optional) :
                аргументы функции residual_h.

        Returns:
            None
        '''
        if self.predict_stage == True and self.multi_update == False:
            print("Error! Double call of update!")
            return
        if self.n >= self.N:
            print("Error! No more data!")
            return
        
        self.z = self.m[self.n, :]

        # если есть args, запаковываем его, чтобы использовать потом *
        if not isinstance(args_j, tuple):
            args_j = (args_j,)

        if not isinstance(args_h, tuple):
            args_h = (args_h,)
            
        # считаем якобиан
        self.H = self.HJacobian(self.x, *args_j)
        
        # S = H*P_prior*H^T + R               
        self.S = self.H @ self.P @ self.H.T + self.R  
        
        # K = P_prior*H^T*S^(-1)
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

        # y = m - H*X_prior        
        self.y = self.residual_h(self.z, self.h(self.x, *args_h))
        self.r = self.y

        # X_post = X_prior + K * y         
        self.x = self.x + self.K @ self.y

        # P_post = (I - K*H) * P_prior     
        self.P = self.P - self.K @ self.H @ self.P
        
        # self._count_parameters()
        
        self.n += 1
        
        self.predict_stage = True
        return
    
    def _first_step(self):
        '''
        Если на первом шаге фильтра не задано при инициализации предсказание,
        делает его равным первому измерению

        Returns
        -------
        None.

        '''
        if self.n == 0 and self.m is not None:
            first_m : np.array(float) = self.m[0]
            
            if self.x is None:    
                self.x = self.h(first_m)
                
            self.prior = np.array(self.x, ndmin=2)
            #self.P_prior = self.P
                                
        elif self.n != 0:
            print("Error! It is not the first step!")
        else:
            print("Error! Array m is empty!")
            
        return
    
    def step(self):
        '''
        Шаг фильтра, предсказание + обновление. Также обновляет знвчение dt, 
        если используется dt_arr

        Returns
        -------
        None.

        '''
        if self.n >= self.N:
            print("Error! No more data!")
            return
        if self.dt_arr is not None: 
            if self.n == 0: self.dt = self.dt_arr[self.n]
            else:           self.dt = self.dt_arr[self.n] - self.dt_arr[self.n - 1]
        self.predict()
        self.update()
        
        return
    
    def calculate_all(self):
        '''
        Запускает фильтр до тех пор, пока не закончатся измерения.

        Returns
        -------
        None.

        '''
        if self.m is None:
            print("Error! Array m is empty!")
            return
            
        for i in range(self.n, self.N):
            self.step()
                
        return
    
 