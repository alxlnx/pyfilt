#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:13:50 2022

@author: yaroslavivashkevich
"""

import numpy as np
from scipy.linalg import cholesky

from FilterLib.Kalman import Kalman



class UKF(Kalman):
    '''
        Сигма-точечный фильтр Калмана

        Args:
        dim_x (int): количество переменных состояния 
        dim_m (int): количество переменных измерения


        P (NDArray(float, ndmin=2)) : матрица ковариации текущего состояния
        f (Callable)                : функция перехода состояния
        Q (NDArray(float, ndmin=2)) : ковариационная матрица шума процесса (переходная ковариация)
        h (Callable)                : матрица измерения
        R (NDArray(float, ndmin=2)) : матрица коварации для шума измерений                                  
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
	    _sigma_point_computation : вычислить сигма-точки
        _weight_computation : вычислить веса сигма-точки

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
    def __init__(self, dim_x : int, dim_m : int, f, h,
                 P=None, Q=None, B=None, u=None,
                 R=None, x0=None ,m=None, dt=None, dt_arr=None,
                 alpha=None, beta=None, kappa=None,
                 residual_x=None, residual_h=None,
                 mean_x=None, mean_z=None):
        # dim_x - количество переменных состояния
        self.dim_x : int = dim_x
        # dim_m - количество переменных измерения
        self.dim_m : int = dim_m

        # P - матрица ковариации текущего состояния
        self.P : np.array(float, ndmin=2) = np.eye(dim_x)
        # f - функция перехода состояния (Xn = F*X)
        self.f = None # function(x(dim_x), dt) return x(dim_x)
        # Q - ковариационная матрица шума процесса (переходная ковариация)
        self.Q : np.array(float, ndmin=2) = np.eye(dim_x)
        # H - функция измерения (y = z - H*x, y - невязка, z - измерение, x - априорное состояние)
        self.h = None # function(x(dim_x)) return x(dim_z)
        # R - матрица коварации для шума измерений
        self.R : np.array(float, ndmin=2) = np.eye(dim_m)

        # K - kalman gain, содержит значения от 0 до 1
        self.K : np.array(float, ndmin=2) = np.zeros((dim_x, dim_m))
        # y - невязка
        self.y : np.array(float, ndmin=1) = np.zeros(dim_m)

        # P_prior - значения предсказанной ковариационной матрицы,       dim=(x, x)
        self.P_prior     : np.array(float, ndmin=2) = None
        # P_posterior - значения отфильтрованной ковариационной матрицы, dim=(x, x)
        self.P_posterior : np.array(float, ndmin=2) = None

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

        # необходимо только для рассчета функции правдоподобия
        # H - матрица измерения (y = z - H*x, y - невязка, z - измерение, x - априорное состояние)
        self.H : np.array(float, ndmin=2) = np.zeros((dim_m, dim_x))
        self.S = None

        # n - шаг
        self.n  : int = 0
        # N - шагов всего
        self.N  : int = 0
        # dt - шаг времени
        self.dt : float = 1
        # dt_arr - массив времени
        self.dt_arr = None

        #сигма-точки
        self.sigmas : np.array(float, ndmin=2) = np.zeros((2 * dim_x + 1, dim_x))
        # здесь хранятся сигма-точки, прогнанные через f
        self.Y      : np.array(float, ndmin=2) = np.zeros((2 * dim_x + 1, dim_x))
        # здесь хранятся сигма-точки, переведенные из размерности dim_x в dim_m
        self.Z      : np.array(float, ndmin=2) = np.zeros((2 * dim_x + 1, dim_m))
        # аналог S
        self.Pz     : np.array(float, ndmin=2) = np.zeros((dim_m, dim_m))

        # веса значений
        self.Wm : np.array(float, ndmin=1) = np.zeros(2 * dim_x + 1)
        # веса ковариаций
        self.Wc : np.array(float, ndmin=1) = np.zeros(2 * dim_x + 1)

        # параметры сигма-точек
        self.alpha      : float = 0.5
        self.beta       : float = 1
        self.kappa      : float = 3 - dim_x
        self.lambda_    : float = None

        # функции разности векторов
        self.residual_x = np.subtract # function
        self.residual_h = np.subtract # function

        # функции вычисления среднего значения
        self.mean_x = np.dot # function(Wm, sigmas)
        self.mean_z = np.dot # function(Wm, sigmas)

        #защита от пвоторного использования update() и predict()
        self.predict_stage : bool = True

        if P  is not None: self.P = P
        if f  is not None: self.f = f
        else :
            print("Error! No f!")
            return
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
        if alpha is not None: self.alpha = alpha
        if beta  is not None: self.beta = beta
        if kappa is not None: self.kappa = kappa
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        if residual_x is not None: self.residual_x=residual_x
        if residual_h is not None: self.residual_h=residual_h
        if mean_x is not None: self.mean_x=mean_x
        if mean_z is not None: self.mean_z=mean_z

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
                self.x = self.h(first_m)

            self.prior = np.array(self.x, ndmin=2)
            self.P_prior = self.P

        elif self.n != 0:
            print("Error! It is not the first step!")
        else:
            print("Error! Array m is empty!")

        return


    def predict(self, **args):
        '''
        Выполнить этап предсказания
        
        $$ Y = f(sigmas) $$
        $$ x_prior = ∑Wm * Y $$
        $$ P_prior = ∑(Wc * (Y - x_prior) * (Y - x).T) + Q $$

        Если это первый шаг фильтра, также вызывает функцию _first_step

        Args:
        args (arguments, optional)
            : аргументы функции f.

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

        # генерируем сигмы, разбиваем одну точку на несколкько сигма-точек
        self._sigma_point_computation()
        # генерируем веса
        self._weight_computation()

        # пропускаем сигмы черерз f
        # Y = f(sigmas)
        for i in range(2 * self.dim_x + 1):
            self.Y[i] = self.f(self.sigmas[i], self.dt, **args)

        # Unscented Transform
        # x_prior = ∑Wm * Y
        self.x = np.zeros(self.dim_x)
        self.x = self.mean_x(self.Wm, self.Y)

        # P_prior = ∑(Wc * (Y - x_prior) * (Y - x).T) + Q
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(2 * self.dim_x + 1):
            dx = self.residual_x(self.Y[i], self.x)
            self.P += self.Wc[i] * np.outer(dx, dx)
        self.P += self.Q

        self.prior = np.append(self.prior, [self.x], axis=0)

        self.P_prior = self.P

        self.predict_stage = False
        return


    def update(self, **args):
        '''
        Вэполнить этап обновления

        $$ Z = h(Y) $$
        $$ nu = ∑Wm * Z $$
        $$ Pz = ∑(Wc * (Z - nu) * (Z - nu).T) + R $$
        $$ y = m - nu $$
        $$ K = (∑(Wc * (Y - x) * (Z - nu).T)) * Pz^(-1) $$
        $$ x_post = x_prior + K * y $$
        $$ P_post = P_prior - K * Pz * K.T $$

        Args:
        args (arguments, optional)
            : аргументы функции h.

        Returns:
        None

        '''
        if self.predict_stage == True:
            print("Error! Double call of update!")
            return
        if self.n >= self.N:
            print("Error! No more data!")
            return

        self.z = self.m[self.n, :]

        # Z = h(Y)
        for i in range(2 * self.dim_x + 1):
            self.Z[i] = self.h(self.Y[i], right=False)

        # Unscented Transform
        # nu = ∑Wm * Z
        self.nu = np.zeros(self.dim_m)
        self.nu = self.mean_z(self.Wm, self.Z)

        # Pz = ∑(Wc * (Z - nu) * (Z - nu).T) + R
        self.Pz = np.zeros((self.dim_m, self.dim_m))
        for i in range(2 * self.dim_x + 1):
            dh = self.residual_h(self.Z[i], self.nu)
            self.Pz += self.Wc[i] * np.outer(dh, dh)
        self.Pz += self.R

        # y = m - nu
        self.y = self.residual_h(self.z, self.nu)
        self.r = self.y

        # K = (∑(Wc * (Y - x) * (Z - nu).T)) * Pz^(-1)
        self.K : np.array(float, ndmin=2) = np.zeros((self.dim_x, self.dim_m))
        for i in range(2 * self.dim_x + 1):
            dx = self.residual_x(self.Y[i], self.x)
            dh = self.residual_h(self.Z[i], self.nu)
            self.K += self.Wc[i] * np.outer(dx, dh)
        self.K = self.K @ np.linalg.inv(self.Pz)

        # x_post = x_prior + K * y
        self.x += self.K @ self.y

        # P_post = P_prior - K * Pz * K.T
        self.P -= self.K @ self.Pz @ self.K.T


        # вычисление функции правдоподобия
        for i in range(self.dim_m):
            if self.x[i * self.dim_m] == 0 or self.nu[i] == 0:
                self.H[i, i * self.dim_m] = 1
            else:
                self.H[i, i * self.dim_m] = self.nu[i]/self.x[i * self.dim_m]


        self._count_parameters()

        self.n += 1

        self.predict_stage = True
        return

    def _sigma_point_computation(self):
        '''
        Вычислить сигма-точки.

        Returns:
        None.

        '''

        #разложение матрицы методом Холецкого
        sqrt = cholesky((self.dim_x + self.lambda_) * self.P)

        self.sigmas[0] = self.x
        for i in range(0, self.dim_x):
            #вычитание массивами
            self.sigmas[i+1]            = self.residual_x(self.x, -sqrt[i])
            self.sigmas[self.dim_x+i+1] = self.residual_x(self.x,  sqrt[i])

        return

    def _weight_computation(self):
        '''
        Вычислить веса сигма-точки
        
        Returns:
        None

        '''
        self.Wm[0] = self.lambda_ / (self.lambda_ + self.dim_x)
        self.Wc[0] = self.Wm[0] + 1 - (self.alpha**2) + self.beta

        for i in range(1, self.dim_x * 2 + 1):
            self.Wm[i] = 1 / (2 * (self.lambda_ + self.dim_x))
            self.Wc[i] = self.Wm[i]

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
            if self.n == 0: self.dt = self.dt_arr[self.n]
            else:           self.dt = self.dt_arr[self.n] - self.dt_arr[self.n - 1]
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
