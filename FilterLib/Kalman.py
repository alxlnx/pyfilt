#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:10:15 2022

@author: yaroslavivashkevich
"""

import numpy as np
import matplotlib.pyplot as plt

class Kalman:
    ''' Обычный фильтр Калмана
    
    Args:
        dim_x (int): количество переменных состояния 
        dim_m (int): количество переменных измерения
        
        
        P (NDArray(float, ndmin=2)) : матрица ковариации текущего состояния
        F (NDArray(float, ndmin=2)) : функция перехода состояния (Xn = F*X)
        Q (NDArray(float, ndmin=2)) : ковариационная матрица шума процесса (переходная ковариация)
        B (NDArray(float, ndmin=2)) : control function (dx = B*u)
        u (NDArray(float, ndmin=1)) : control input
        H (NDArray(float, ndmin=2)) : матрица измерения (y = z - H*x, y - невязка, z - измерение, x - априорное состояние)                      
        R (NDArray(float, ndmin=2)) : матрица коварации для шума измерений                                  
        S (NDArray(float, ndmin=2)) : system uncertainty (общий шум)
        K (NDArray(float, ndmin=2)) : kalman gain, содержит значения от 0 до 1        
        y (NDArray(float, ndmin=1)) : невязка
    
        x (NDArray(float, ndmin=1)) : текущем состоянии, dim=(x, 1), при инициализации - предсказание начального положения
        z (NDArray(float, ndmin=1)) : данные о текущем измерении, dim=(x, 1)
        m (NDArray(float, ndmin=2)) : значения измерений,         dim=(m, 1)
        
        prior       (NDArray(float, ndmin=2)) : значения предсказаний,  dim=(x, 1)
        posterior   (NDArray(float, ndmin=2)) : значения фильтра, dim=(x, 1)
        P_prior     (NDArray(float, ndmin=3)) : значения предсказанной ковариационной матрицы,   dim=(x, x)
        P_posterior (NDArray(float, ndmin=3)) : значения отфильтрованной ковариационной матрицы, dim=(x, x)

        res_arr     (NDArray(float, ndmin=2)) : массив невязок
        l_arr       (NDArray(float, ndmin=1)) : массив функций правдоподобия
        e_arr       (NDArray(float, ndmin=1)) : массив эпсилон (невязка в квадрате, деленная на общий шум)
        r           (NDArray(float, ndmin=1)) : невязка в декартоввых координатах

        n           (int) : шаг
        N           (int) : шагов всего
        dt          (float) : шаг времени
        dt_arr      (NDArray(float, ndmin=1)) : массив времени, если None, то всегда используется dt

        predict_stage (bool) : защита от пвоторного использования update() и predict()
        
    Methods:
        add_measurement   : добавить измернеия
        predict           : выполнить этап предсказания
        update            : выполнить этап обновления
        step              : сделать шаг фильтра, предсказание + обновление
        calculate_all     : запустить фильтр до конца массива измерений
        likelihood        : рассчитать функцию правдоподобия
        epsilon           : рассчитать невязку в квадрате, деленная на общий шум
        
        _first_step       : выполнить первый шаг фильтра
        _count_parameters : вычислить и сохранить различные парамаетры
        
        save_graphic      : сохранить графики задынных данных
        P_medium          : Среднее значение главной диагонали матрицы P
        plot_p            : построить и сохранить график средних значений диагонали матрицы ковариации
        plot_r            : построить и сохранить график невязки
        plot_r3           : построить и сохранить график абсолютной невязки по всем координатам
        plot_l            : построить и сохранить график значений функции правдоподобия
        plot_e            : построить и сохранить график эпсилон
        plot_3d           : построить и сохранить график отфильтрованных значений в 3d
        plot_compare_res  : построить и сохранить график разницы между отфильтрованными и заданными значениями
    '''
    
    
    
    def __init__(self, dim_x : int, dim_m : int, 
                 P=None, F=None, Q=None, B=None, u=None, 
                 H=None, R=None, x0=None,m=None, dt=None, dt_arr=None):
        
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
        # H - функция измерения (y = z - H*x, y - невязка, z - измерение, x - априорное состояние)                      
        self.H : np.array(float, ndmin=2) = np.zeros((dim_m, dim_x))
        # R - матрица коварации для шума измерений                                  
        self.R : np.array(float, ndmin=2) = np.eye(dim_m)
       
        # S - system uncertainty (общий шум)
        self.S : np.array(float, ndmin=2) = np.zeros((dim_m, dim_m)) 
        # K - kalman gain, содержит значения от 0 до 1
        self.K : np.array(float, ndmin=2) = np.zeros((dim_x, dim_m)) 
        # y - невязка
        self.y : np.array(float, ndmin=1) = np.zeros(dim_m) 
                 
        # P_prior - значения предсказанной ковариационной матрицы,       dim=(x, x)
        self.P_prior     : np.array(float, ndmin=3) = None
        # P_posterior - значения отфильтрованной ковариационной матрицы, dim=(x, x)
        self.P_posterior : np.array(float, ndmin=3) = None
        
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
        self.n : int = 0
        # N - шагов всего
        self.N : int = 0
        # dt - шаг времени
        self.dt : float = dt
        # dt_arr - массив времени
        self.dt_arr = dt_arr

        #защита от пвоторного использования update() и predict()
        self.predict_stage : bool = True
        
        if P  is not None: self.P = P
        if F  is not None: self.F = F
        if Q  is not None: self.Q = Q
        if B  is not None: self.B = B
        if u  is not None: self.u = u
        if H  is not None: self.H = H            
        if R  is not None: self.R = R
        if x0 is not None: self.x = x0
        if dt_arr is not None: self.dt_arr= dt_arr
        if dt is not None: self.dt= dt
        if m  is not None: 
            self.m = m           
            self.N = len(m)
            
        return
    

    
    def add_measurement(self, data : np.array(float, ndmin=2), time : np.array(float, ndmin=2) = None):                
        '''
        Добавить новые измерения

        Args:
        data (NDArray(float), optional) 
            : массив значений, добавляемый к массиву измерений
        time (NDArray(float), optional) 
            : массив значений, добавляемый к массиву времени
            
        Returns:
        None

        '''
        
        if self.m is None:
            self.m = data
            self.N = len(data)
        else:
            self.m = np.append(self.m, data, axis=0)
            self.N += len(data)
        
        if time is not None:
            if self.dt_arr() is None:
                self.dt_arr = time
            else:
                self.dt_arr = np.append(self.dt_arr, time, axis=0)
        
        return
    
    def set_control_input(self, u : np.array(float, ndmin = 1)):
        self.u = u
        return
    
    def get_prediction(self):
        return self.prior
        
    def get_measurement(self):
        return self.m
    
    def get_filter(self):
        return self.posterior
        
    def predict(self):   
        '''
        Выполнить этап предсказания
        
        $$ X_prior = F*X_post + B*u $$
        $$ P_prior = F*P_post*F^T + Q $$

        Если это первый шаг фильтра, также вызывает функцию _first_step

        Returns:
        None

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

        self.prior   = np.append(self.prior,   [self.x], axis=0)
        self.P_prior = np.append(self.P_prior, [self.P], axis=0)
        
        self.predict_stage = False
        return
        
    def update(self):
        '''
        Вэполнить этап обновления
        
        $$ S = H*P_prior*H^T + R $$ 
        $$ K = P_prior*H^T*S^(-1) $$    
        $$ y = m - H*X_prior $$
        $$ X_post = X_prior + K * y $$
        $$ P_post = (I - K*H) * P_prior $$
        
        Также вычисляются и сохраняются такие параметры, как невязка и функция правдоподобия
        
        Returns
        -------
        None.

        '''
        if self.predict_stage == True:
            print("Error! Double call of update!")
            return
        if self.n >= self.N:
            print("Error! No more data!")
            return
        
        self.z = self.m[self.n, :]
        
        # S = H*P_prior*H^T + R               
        self.S = self.H @ self.P @ self.H.T + self.R  
        
        # K = P_prior*H^T*S^(-1)           
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

        # y = m - H*X_prior                
        self.y = self.z - self.H @ self.x
        
        # X_post = X_prior + K * y         
        self.x = self.x + self.K @ self.y

        # P_post = (I - K*H) * P_prior     
        self.P = (np.eye(self.dim_x) - self.K @ self.H) @ self.P
        
        self.r = self.y
        self._count_parameters()
        
        self.n += 1
        
        self.predict_stage = True
        return
    
    def _first_step(self):
        '''
        Выполнить первый шаг
        Если на первом шаге фильтра не задано при инициализации предсказание,
        делает его равным первому измерению

        Returns:
        None

        '''
        
        if self.n == 0 and self.m is not None:
            first_m : np.array(float) = self.m[0]
            
            if self.x is None:    
                self.x = first_m @ self.H
                
            self.prior   = np.array(self.x, ndmin=2)
            self.P_prior = np.array(self.P, ndmin=3)
                   
        elif self.n != 0:
            print("Error! It is not the first step!")
        else:
            print("Error! Array m is empty!")
            
        return
    
    def step(self):
        '''
        Выполнить шаг фильтра, предсказание + обновление. Также обновляет знвчение dt, 
        если используется dt_arr

        Returns:
        None

        '''
        if self.n >= self.N:
            print("Error! No more data!")
            return
        if self.dt_arr is not None: 
            if self.n == 0 and self.dt is None and self.n + 1 >= self.N: 
                self.dt = self.dt_arr[self.n + 1] - self.dt_arr[self.n]
            elif self.n == 0 and self.dt is not None: 
                self.dt = self.dt
            else: 
                self.dt = self.dt_arr[self.n] - self.dt_arr[self.n - 1]
        self.predict()
        self.update()
        
        return
    
    def calculate_all(self):
        '''
        Запустиь фильтр до тех пор, пока не закончатся измерения.

        Returns:
        None

        '''
        if self.m is None:
            print("Error! Array m is empty!")
            return
            
        for i in range(self.n, self.N):
            self.step()
                
        return
    
    def likelihood(self, P, H, R):
        '''
        Вычислить функцию правдоподобия
        $$ likelihood = exp(-0.5 * y.T @ S^(-1) @ y) $$

        Parameters:
        P (NDArray(float, ndmin=2))
            : матрица ковариации текущего состояния.
        H (NDArray(float, ndmin=2))
            : функция измерения.
        R (NDArray(float, ndmin=2))
            : матрица коварации для шума измерений.

        Returns:
        likelihood (float)
            : значение функции правдоподобия.

        '''
        S = np.dot(H, np.dot(P, H.T)) + R
        likelihood = np.exp(-0.5 * self.y.T @ np.linalg.inv(S) @ self.y)
        return likelihood
    
    def epsilon(self, P, H, R):
        '''
        Вычислить эпсилон - невязка в квадрате, деленная на общий шум
        $$ e = y.T @ S^(-1) @ y $$

        Args:
        P (NDArray(float, ndmin=2))
            : матрица ковариации текущего состояния.
        H (NDArray(float, ndmin=2))
            : функция измерения.
        R (NDArray(float, ndmin=2))
            : матрица коварации для шума измерений.

        Returns:
        e (float)
            : эпсилон.

        '''
        S = H @ P @ H.T + R
        e = self.y.T @ np.linalg.inv(S) @ self.y
        return e
    
    def _count_parameters(self):
        '''
        Вычислить и сохранить различные парамаетры, а именно, апостериорные 
        значения x и P, невязки, значения эпсилон и функции правдоподобия.


        Returns:
        None

        '''
        if self.n != 0:
            if  np.size(self.x.shape) == 2:
                self.posterior   = np.append(self.posterior,   self.x,  axis=0)
                self.P_posterior = np.append(self.P_posterior, self.P,  axis=0)
                self.res_arr     = np.append(self.res_arr,     self.r,  axis=0)
                self.l_arr       = np.append(self.l_arr,       self.likelihood(self.P, self.H, self.R),  axis=0)
                self.e_arr       = np.append(self.e_arr,       self.epsilon(self.P, self.H, self.R),  axis=0)
            else:
                self.posterior   = np.append(self.posterior,  [self.x], axis=0)
                self.P_posterior = np.append(self.P_posterior,[self.P], axis=0)
                self.res_arr     = np.append(self.res_arr,    [self.r], axis=0)
                self.l_arr       = np.append(self.l_arr,      [self.likelihood(self.P, self.H, self.R)], axis=0)
                self.e_arr       = np.append(self.e_arr,      [self.epsilon(self.P, self.H, self.R)], axis=0)
        else:
            self.posterior   = np.array(self.x, ndmin=2)
            self.P_posterior = np.array(self.P, ndmin=3)
            self.res_arr     = np.array(self.r, ndmin=2)
            self.l_arr       = np.array(self.likelihood(self.P, self.H, self.R), ndmin=1)
            self.e_arr       = np.array(self.epsilon(self.P, self.H, self.R), ndmin=1)
        return
    
    def save_graphic(self, data, path : str, name : str):
        '''
        Сохраненить массив data в txt файл

        Args:
        data (NDArray(float, ndmin=1))
            : Сохраняемые данные
        path (str)
            : Путь к файлу
        name (str)
            : Название файла

        Returns:
        None

        '''
        if name is None: 
            print("Error! No save_name!")
            return
        
        name = path + name + ".txt"
        f = open(name, 'w')
        i = 0
        for d in data:
            if (i != 0):
                f.write(str(d) + '\n')
            i += 1
        print("Data saved in " + name)
        f.close()
        return 
    
    def P_medium(self):
        '''
        Вычислить среднее значения главной диагонали матрицы P

        Returns:
        p_sqr (float)
            : Среднее значение главной диагонали матрицы P.

        '''
        p_diag = np.zeros((self.N, self.dim_x))
        p_sqr  = np.zeros(self.N)
        
        for i in range(len(self.P_posterior[:, 0, 0])):
            #print(self.P_posterior[i, :, :])
            p_diag[i] = np.diag(self.P_posterior[i, :, :])
            p_sqr[i] = np.sqrt(p_diag[i, 0] ** 2 + p_diag[i, 3] ** 2 + p_diag[i, 6] ** 2)
            
            #for ii in range(len(p_diag[i])): 
            #    p_sqr[i] += p_diag[i, ii] ** 2
            #p_sqr[i] = np.sqrt(p_sqr[i])
            
            
        return p_sqr
    
    def plot_p(self, loc=4, name=None, save_name=None, plot=True, **kwargs):
        '''
        Построить и сохранить график средних значений диагонали матрицы ковариации

        Args:
        loc (int, optional)
            : расположение легендыы на графике. По умолчанию 4.
        name (str, optional)
            : Название графика. По умолчанию None.
        save_name (str, optional)
            : Назввание файла для сохранения. По умолчанию None.
        plot (bool, optional)
            : Если True - рисует и сохраняет график, иначе возвращает массив 
              средних значений диагоналей матриц ковариации. По умолчанию True.
        **kwargs (keyword arguments)
            : опциональные аргументы для функции plot.
            
        Returns:
        aver (np.array(float, ndmin=1))
            : массив средних значений диагоналей матриц ковариации.

        '''
        
        #сейчас только на y
        plot_y = np.array(self.P_medium())
        plot_x = np.arange(0, len(plot_y)*self.dt, self.dt)
        aver = np.average(plot_y)

        if plot is True:
            fig, ax = plt.subplots()
            fig = plt.plot(plot_x, plot_y, color='blue', label='p', **kwargs)

            plt.legend(loc=loc)
            plt.grid()
            if name is None: 
                ax.set_title("P")
            else:
                ax.set_title("P по " + name)
            if save_name is not None: self.save_graphic(plot_y, save_name)
        else:
            return aver

        return
    
    
    def plot_r(self, axis_h=0, loc=4, name=None, save_name=None, **kwargs):        
        '''
        Построить и сохранить график невязки

        Args:
        axis_h (int, optional)
            : Номер координаты, по которой строится нгевязка. По умолчанию 0.
        loc (int, optional)
            : расположение легендыы на графике. По умолчанию 4.
        name (str, optional)
            : Название графика. По умолчанию None.
        save_name (str, optional)
            : Назввание файла для сохранения. По умолчанию None.
        **kwargs (keyword arguments)
            : опциональные аргументы для функции plot.

        Returns:
        None

        '''
        plot_y = np.array(self.res_arr[:, axis_h])
        plot_x = np.arange(0, len(plot_y)*self.dt, self.dt)

        fig, ax = plt.subplots()
        fig = plt.plot(plot_x, plot_y, color='blue', label='residual', **kwargs)
                        
        plt.legend(loc=loc)
        plt.grid()
        if name is None:
            ax.set_title("Невязка")
        else:
            ax.set_title("Невязка " + name)
            
        if save_name is not None: self.save_graphic(plot_y, save_name)
        

        return 
    
    
    def plot_r3(self, loc=4, name=None, save_name=None, **kwargs): 
        '''
        Построить и сохранить график абсолютной невязки по всем координатам

        Args:
        loc (int, optional)
            : расположение легендыы на графике. По умолчанию 4.
        name (str, optional)
            : Название графика. По умолчанию None.
        save_name (str, optional)
            : Назввание файла для сохранения. По умолчанию None.
        **kwargs (keyword arguments)
            : опциональные аргументы для функции plot.

        Returns:
        None

        '''
        plot_y = np.array(np.sqrt(self.res_arr[:, 0]**2 + 
                                  self.res_arr[:, 1]**2 + 
                                  self.res_arr[:, 2]**2))
        plot_x = np.arange(0, len(plot_y)*self.dt, self.dt)

        fig, ax = plt.subplots()
        fig = plt.plot(plot_x, plot_y, color='blue', label='residual', **kwargs)
                        
        plt.legend(loc=loc)
        plt.grid()
        if name is None:
            ax.set_title("Невязка")
        else:
            ax.set_title("Невязка по " + name)
            
        if save_name is not None: self.save_graphic(plot_y, save_name)
        

        return 
        
    
    def plot_l(self, loc=4, name=None, save_name=None, **kwargs):  
        '''
        Построить и сохранить график значений функции правдоподобия.

        Args:
        loc (int, optional)
            : расположение легендыы на графике. По умолчанию 4.
        name (str, optional)
            : Название графика. По умолчанию None.
        save_name (str, optional)
            : Назввание файла для сохранения. По умолчанию None.
        **kwargs (keyword arguments)
            : опциональные аргументы для функции plot.
            
        Returns:
        None

        '''
        plot_y = np.array(self.l_arr)
        plot_x = np.arange(0, len(plot_y)*self.dt, self.dt)
        
        fig, ax = plt.subplots()
        fig = plt.plot(plot_x, plot_y, color='blue', label='likelihood', **kwargs)

        plt.legend(loc=loc)
        plt.grid()
        if name is None: 
            ax.set_title("Правдоподобие")
        else:
            ax.set_title("Невязка по " + name)
        if save_name is not None: self.save_graphic(plot_y, save_name)

        return    
    
    def plot_e(self, loc=4, name=None, save_name=None, **kwargs):     
        '''
        Построить и сохранить график значений эпсилон.

        Args:
        loc (int, optional)
            : расположение легендыы на графике. По умолчанию 4.
        name (str, optional)
            : Название графика. По умолчанию None.
        save_name (str, optional)
            : Назввание файла для сохранения. По умолчанию None.
        **kwargs (keyword arguments)
            : опциональные аргументы для функции plot.

        Returns:
        None

        '''
        plot_y = np.array(self.e_arr)
        plot_x = np.arange(0, len(plot_y)*self.dt, self.dt)
        
        fig, ax = plt.subplots()
        fig = plt.plot(plot_x, plot_y, color='blue', label='epsilon', **kwargs)

        plt.legend(loc=loc)
        plt.grid()
        if name is None:
            ax.set_title("Невязка в квадрате, деленная на матрицу S")
        else:
            ax.set_title("Невязка по " + name)
        if save_name is not None: self.save_graphic(plot_y, save_name)

        return 
    
    def plot_3d(self, dim=3, radar_pos=None):  
        '''
        Построить и сохранить 3d график значений фильтра.

        Args:
        dim (int)
            : Колтчество осей координат. По умолчанию 3.
        radar_pos (np.array(float, ndmin=1), optional), [x, y, z], 
            : Координаты радиолокатора в МЗСК.  По умолчанию None.

        Returns:
        None

        '''
        dim = self.dim_x
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lim_z = self.posterior[int(len(self.posterior[:, 0])/2)][dim]
        ax.set_xlim(self.posterior[0][0]/1000, self.posterior[-1][0]/1000); ax.set_ylim(self.posterior[0][2*dim]/1000, self.posterior[-1][2*dim]/1000); ax.set_zlim(0, lim_z/1000);
        ax.scatter(self.posterior[:, 0]/1000, self.posterior[:, 2*dim]/1000, self.posterior[:, dim]/1000, label='parametric curve', color='red')
        if radar_pos is not None:
            ax.scatter(radar_pos[0]/1000, radar_pos[2]/1000, radar_pos[1]/1000, label='parametric curve', color='orange')
        ax.set_xlabel('km')
        ax.set_ylabel('km')
        ax.set_zlabel('km')
        ax.set_title("Отфильтрованная траектория")
        
        return
    
    def plot_compare_res(self, new_data, name=None, loc=4, save_name=None, plot=True, **kwargs):
        '''
        Построить и сохранить график разницы между отфильтрованными и заданными значениями

        Args:
        new_data (np.array(float, ndmin=1))
            : Массив данных, с которым будут сравнены отфильтрованные значения.
        loc (int, optional)
            : расположение легендыы на графике. По умолчанию 4.
        name (str, optional)
            : Название графика. По умолчанию None.
        save_name (str, optional)
            : Назввание файла для сохранения. По умолчанию None.
        plot (bool, optional)
            : Если True - рисует и сохраняет график, иначе возвращает массив 
              средних значений диагоналей матриц ковариации. По умолчанию True.
        **kwargs (keyword arguments)
            : опциональные аргументы для функции plot.

        Returns:
        aver (np.array(float, ndmin=1))
            : Среднее значение итогового массива.

        '''
        dim = int(self.dim_x / 3)
        filter_data = np.array([self.posterior[:, 0], self.posterior[:, dim], self.posterior[:, 2*dim]]).T

        new_data = np.delete(new_data, 0, axis=0)
        residual = filter_data - new_data
        
        plot_y = np.array(np.sqrt(residual[:, 0]**2+residual[:, 1]**2+residual[:, 2]**2))
        plot_x = np.arange(0, len(plot_y)*self.dt, self.dt)
        
        aver = np.average(plot_y)

        if plot is True:
            fig, ax = plt.subplots()
            fig = plt.plot(plot_x, plot_y, color='blue', label='residual', **kwargs)
                                 
            plt.legend(loc=loc)
            plt.grid()
            
            if name is None:
                ax.set_title("Невязка между симуляторными и отфильтрованными данными")
            else:
                ax.set_title("Невязка между симуляторными и отфильтрованными данными по" + name)  


            if save_name is not None: self.save_graphic(plot_y, save_name)
        else:
            return aver
        
        return 
    
