# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:28:49 2023

@author: y.ivashkevich
"""


import numpy as np

class ReInit():
    '''
        Фильтр Калмана с переназначением параметров.

        Args:
        f              (filter)                   : фильтр
        Q_scale_factor (float)                    : во сколько раз меняется Q
        eps_max        (float)                    : порог эпсилон, после которого начинают меняться параметры
        count          (int)                      : сколько итераций подряд мы изменяем параметры
        X              (np.array(float, ndmin=2)) : массив значений фильтра
        eps            (np.array(float, ndmin=2)) : массив epsilon фильтра

        Methods:
        step           : шаг фильтра, предсказание + обновление
        run            : запускает фильтр до конца массива измерений
    '''
    def __init__(self, f, Q_scale_factor : float, eps_max : float):
        # фильтр
        self.f = f
        # во сколько раз меняется Q
        self.Q_scale_factor = Q_scale_factor
        # порог эпсилон, после которого начинают меняться параметры
        self.eps_max = eps_max
        # сколько итераций подряд мы изменяем параметры
        self.count = 0
        # массив значений фильтра
        self.X = []
        # массив epsilon фильтра
        self.eps = []


    def step(self, args_pred=(), args_upd=()):
        '''
        Выполнить шаг фильтра, предсказание + обновление. Также обновляет знвчение dt,
        если используется dt_arr. Если epsilon больше определенного знаения, то
        увеличивает матрицу Q, иначе уменьшает до исходного значения.

        Args:
        args_pred (arguments, optional)
            : аргументы функции predict.
        args_upd  (arguments, optional)
            : аргументы функции update.
            
        Returns:
        None

        '''
        # если есть args, запаковываем его, чтобы использовать потом 
        if not isinstance(args_pred, tuple):
            args_pred = (args_pred,)

        if not isinstance(args_upd, tuple):
            args_upd = (args_upd,)

        if self.f.dt_arr is not None: 
            if self.f.n == 0: self.f.dt = self.f.dt_arr[self.f.n]
            else:           self.f.dt = self.f.dt_arr[self.f.n] - self.f.dt_arr[self.f.n - 1]
            
        self.f.predict(*args_pred)
        self.f.update(*args_upd)
        epsilon = self.f.epsilon(self.f.P, self.f.H, self.f.R)
        self.eps.append(epsilon)
        self.X.append(self.f.x)
        if epsilon > self.eps_max:
            self.f.Q *= self.Q_scale_factor
            self.count += 1
        elif self.count > 0:
            self.f.Q /= self.Q_scale_factor
            self.count -= 1

        return

    def run(self, args_pred=(), args_upd=()):
        '''
        Запустить фильтр до тех пор, пока не закончатся измерения.

        Args:
        args_pred (arguments, optional)
            : аргументы функции predict.
        args_upd  (arguments, optional)
            : аргументы функции update.
            
        Returns:
        None

        '''

        if not isinstance(args_pred, tuple):
            args_pred = (args_pred,)

        if not isinstance(args_upd, tuple):
            args_upd = (args_upd,)

        for i in range(self.f.N):
            self.step(args_pred=(args_pred), args_upd=(args_upd))
