# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:19:36 2023

@author: y.ivashkevich
"""

import FilterLib.Switch as sw

import numpy as np

class MMAE(sw.Switch):
    ''' 
        Multiple Model Adaptive Estimator - берет фильтры в пропорции по функции правдоподобия
    
        Args:
        filters     (filters) : массив фильтров
        N_filters   (int) : количество фильтров
        x           (NDArray(float, ndmin=1)) : значение 
        X           (NDArray(float, ndmin=2)) : массив значений фильтра
        eps         (NDArray(float, ndmin=1)) : массив epsilon фильтра
        likelihood  (NDArray(float, ndmin=1)) : массив функций правдоподобия
        p           (NDArray(float, ndmin=1)) : массив вероятности перехода
        dim_m       (float) : размерность измерений
        
        res_arr     (NDArray(float, ndmin=2)) : массив невязок
        l_arr       (NDArray(float, ndmin=1)) : массив функций правдоподобия
        e_arr       (NDArray(float, ndmin=1)) : массив эпсилон (невязка в квадрате, деленная на общий шум)
        y           (NDArray(float, ndmin=1)) : невязка
        r           (NDArray(float, ndmin=1)) : невязка в декартоввых координатах
    '''
    def __init__(self, filters, dim_m=3, dt=1):
        # массив фильтров
        self.filters = filters
        # количество фильтров
        self.N_filters = len(filters)
        # значение 
        self.x = None
        # массив значений фильтра
        self.X = None
        # массив epsilon фильтра
        self.eps = None
        # массив функций правдоподобия
        self.likelihood = np.zeros(self.N_filters)
        # массив вероятности перехода
        self.p = np.ones(self.N_filters) / self.N_filters
        # размерность измерений
        self.dim_m = dim_m
        
        # массив невязок
        self.res_arr     : np.array(float, ndmin=2) = None
        # массив функций правдоподобия
        self.l_arr       : np.array(float, ndmin=1) = None  
        # массив эпсилон (невязка в квадрате, деленная на общий шум)
        self.e_arr       : np.array(float, ndmin=1) = None  
        # неввязка
        self.y           : np.array(float, ndmin=1) = None 
        # неввязка в декартоввых координатах
        self.r           : np.array(float, ndmin=1) = None 
        
        
    def step(self, args_pred_ar=(), args_upd_ar=()):
        '''
        Выполнить шаг фильтра, предсказание + обновление.

        Args:
        args_pred_ar (arguments, optional)
            : аргументы функции predict используемых фильтров. 
        args_upd_ar (arguments, optional)
            : аргументы функции update используемых фильтров.

        Returns:
        None

        '''
        if not isinstance(args_pred_ar, tuple):
            args_pred_ar = (args_pred_ar,)

        if not isinstance(args_upd_ar, tuple):
            args_upd_ar = (args_upd_ar,)
            
        for i in range(self.N_filters):
            self.filters[i].predict(*args_pred_ar)
            self.filters[i].update(*args_upd_ar)
            
            self.likelihood[i] = self.filters[i].likelihood(self.filters[i].P, self.filters[i].H, self.filters[i].R) * self.p[i]
        
        # Если функции правдоподобия всех фильтров равны 0, то они приравниваются к 1
        zero = 0 
        for i in range(len(self.filters)): 
            if self.likelihood[i] <= 0: zero += 1
        if zero == len(self.filters): self.likelihood = np.ones(len(self.filters))
        
        
        for i in range(self.N_filters):
            self.p[i] = self.likelihood[i] / (sum(self.likelihood))
    
        self.x = np.zeros(self.filters[0].dim_x)
        self.r = np.zeros(self.filters[0].dim_m)
        self.y = np.zeros(self.filters[0].dim_m)
        for i in range(self.N_filters):    
            self.x += self.p[i] * self.filters[i].x
            #print(self.r, self.p[i], self.filters[i].r)
            #self.r += self.p[i] * self.filters[i].r
            self.y += self.p[i] * self.filters[i].y

        self.compute()

        if self.X is not None:
            self.X   = np.append(self.X, [self.x], axis=0)
        else:
            self.X   = np.array(self.x, ndmin=2)
            
    def run(self, args_pred=(), args_upd=()):
        '''
        Запустить фильтр до тех пор, пока не закончатся измерения.

        Args:
        args_pred (arguments, optional)
            : аргументы функции step используемых фильтров. 
        args_upd (arguments, optional)
            : аргументы функции step используемых фильтров.

        Returns:
        None
        '''
        if not isinstance(args_pred, tuple):
            args_pred = (args_pred,)

        if not isinstance(args_upd, tuple):
            args_upd = (args_upd,)

        for i in range(self.filters[0].N):
            self.step(args_pred_ar=(args_pred), args_upd_ar=(args_upd))        
