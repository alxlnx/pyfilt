# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:19:14 2023

@author: y.ivashkevich
"""

import FilterLib.Switch as sw

import numpy as np

class IMM(sw.Switch):
    '''
        Interacting Multiple Models - множество взаимодействующих моделей.
        
    Args:
        filters     (NDArray(filters, ndmin=1)) : массив фильтров
        Nf          (int) : количество фильтров
        
        n  (int) : номер шага
        
        mu (NDArray(filters, ndmin=1)) : вероятность того, что фильтр корректен
        M  (NDArray(filters, ndmin=2)) : цепь Маркова
        
        x  (NDArray(float, ndmin=1)) : значение 
        P  (NDArray(float, ndmin=2)) : матрица ковариации текущего состояния
        X  (NDArray(float, ndmin=2)) : массив значений фильтра
        
        likelihood (np.zeros(float, ndmin=1)) : массив значений функции правдоподобия

        x_prior (NDArray(float, ndmin=2)) : значения предсказаний,  dim=(x, 1)
        x_post  (NDArray(float, ndmin=2)) : значения фильтра, dim=(x, 1)
        P_prior (NDArray(float, ndmin=3)) : значения предсказанной ковариационной матрицы,   dim=(x, x)
        P_post  (NDArray(float, ndmin=3)) : значения отфильтрованной ковариационной матрицы, dim=(x, x)
        
        res_arr (NDArray(float, ndmin=2)) : массив невязок
        l_arr   (NDArray(float, ndmin=1)) : массив функций правдоподобия
        e_arr   (NDArray(float, ndmin=1)) : массив эпсилон (невязка в квадрате, деленная на общий шум) 
        y       (NDArray(float, ndmin=1)) : неввязка 
        r       (NDArray(float, ndmin=1)) : неввязка в декартоввых координатах 
        
        omega   (NDArray(float, ndmin=2)) : смешанная вероятность
    Methods:
        compute_mixing_probabilities : вычислить смешанную вероятность
        compute_state_estimate       : иннициализировать состояние основанное на текущем фильтре
    '''
    def __init__(self, filters, mu, M, dt=1):
        self.filters : np.array(filter, ndmin=1) = filters
        self.Nf : int = len(filters)
        
        self.n = 0
        
        # вероятность того, что фильтр корректен
        self.mu = np.asarray(mu) / np.sum(mu)
        # цепь Маркова
        self.M = M
        
        self.x : np.array(float, ndmin=1) = filters[0].x
        self.P : np.array(float, ndmin=2) = np.eye(filters[0].dim_x)
        
        # массив значений фильтра
        self.X = None
        
        self.likelihood = np.zeros(self.Nf)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post  = self.x.copy()
        self.P_post  = self.P.copy()

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

        # смешанная вероятность
        self.omega = np.zeros((self.Nf, self.Nf))
        self.compute_mixing_probabilities()
        
        # иннициализировать состояние основанное на текущем фильтре
        self.compute_state_estimate()
        
        return
    
    def compute_mixing_probabilities(self):
        '''
        Вычислить смешанную вероятность

        Returns:
        None

        '''
        
        self.cbar = np.dot(self.mu, self.M)
        for i in range(self.Nf):
            for j in range(self.Nf):
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]
        return
        
    def compute_state_estimate(self):
        '''
        Иннициализировать состояние основанное на текущем фильтре

        Returns:
        None

        '''

        self.x.fill(0)
        for f, mu in zip(self.filters, self.mu):
            self.x = self.x + f.x * mu

        
        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            y = f.x - self.x
            self.P += mu * (np.outer(y, y) + f.P)
        return
    
    def predict(self):
        '''
        Выполнить этап предсказания

        Returns:
        None

        '''
        xs, Ps = [], []
     
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            x = np.zeros(self.x.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj

            xs.append(x)

            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj * (np.outer(y, y) + kf.P)
            Ps.append(P)
        
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].copy()
            f.P = Ps[i].copy()
            f.predict()
         
        # compute mixed IMM state and covariance and save posterior estimate
        self.compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        
        return
        
    def update(self):
        '''
        Вэполнить этап обновления

        Returns:
        None

        '''
        
        for i, f in enumerate(self.filters):
            f.update()
            self.likelihood[i] = f.likelihood(self.filters[i].P, self.filters[i].H, self.filters[i].R)
        zero = 0 
        for i in range(len(self.filters)): 
            if self.likelihood[i] <= 0: zero += 1
        if zero == len(self.filters): self.likelihood = np.ones(len(self.filters))
        #print('likelihood', self.likelihood)

        # update mode probabilities from total probability * likelihood
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        self.compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self.compute_state_estimate()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
        self.r = np.zeros(self.filters[0].dim_m)
        self.y = np.zeros(self.filters[0].dim_m)
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            for kf, wj in zip(self.filters, w):      
                #self.r += kf.r * wj
                self.y += kf.y * wj
        self.compute()
        
        if self.X is not None:
            self.X   = np.append(self.X,  [self.x], axis=0)
        else:
            self.X   = np.array(self.x, ndmin=2)
            
        self.n += 1

        return
       
    def step(self):
        '''
        Выполнить шаг фильтра, предсказание + обновление. Также обновляет знвчение dt, 
        если используется dt_arr

        ReturnsL:
        None

        '''
        self.predict()
        self.update()
        return

    def run(self):
        '''
        Запустиь фильтр до тех пор, пока не закончатся измерения.

        Returns:
        None

        '''
        for i in range(self.N):
            self.step()
        return
