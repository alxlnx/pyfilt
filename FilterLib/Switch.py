# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:19:44 2023

@author: y.ivashkevich
"""

import matplotlib.pyplot as plt
import numpy as np

class Switch():
    '''
        Переключатель фильтров, запускает гребенку фильтров, выбирает значение 
        фильтра с наименьшей невязкой.
        
    Args:
        filters     (NDArray(filters, ndmin=1)) : массив фильтров
        N_filters   (int) : количество фильтров
        eps_max     (float) : порог эпсилон, после которого мы начинаем менять параметры
        x           (NDArray(float, ndmin=1)) : значение 
        X           (NDArray(float, ndmin=2)) : массив значений фильтра
        eps         (NDArray(float, ndmin=1)) : массив epsilon фильтра
        dim_m       (int) : размерность измерений
        res_arr     (NDArray(float, ndmin=2)) : массив невязок
        l_arr       (NDArray(float, ndmin=1)) : массив функций правдоподобия
        e_arr       (NDArray(float, ndmin=1)) : массив эпсилон (невязка в квадрате, деленная на общий шум)
        y           (NDArray(float, ndmin=1)) : неввязка
        r           (NDArray(float, ndmin=1)) : невязка в декартоввых координатах 
        
        dt          (float) : шаг времени 
        dt_arr      (NDArray(float, ndmin=1)) : массив времени, если None, то всегда используется dt

    Methods:
        step              : выполнить шаг фильтра, предсказание + обновление
        run               : запустиь фильтр до конца массива измерений
        
        _compute          : Вычислить и сохранить различные парамаетры.
        
        likelihood        : функция правдоподобия
        epsilon           : невязка в квадрате, деленная на общий шум
        save_graphic      : сохранение графика задынных данных
        plot_r            : построение и сохранение графика невязки
        plot_l            : построение и сохранение графика значений функции правдоподобия
        plot_3d           : построение и сохранение отфильтрованных значений в 3d
        plot_compare_res  : построение и сохранение разницы между отфильтрованными и заданными значениями
    '''
    
    def __init__(self, filters, eps_max, dim_m=3, dt=1, dt_arr=None):
        # массив фильтров
        self.filters = filters
        # количество фильтров
        self.N_filters = len(filters)
        # порог эпсилон, после которого мы начинаем менять параметрыы
        self.eps_max = eps_max
        # значение 
        self.x = None
        # массив значений фильтра
        self.X = None
        # массив epsilon фильтра
        self.eps = None
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
        
        # n - шаг
        self.n : int = 0
        # dt - шаг времени
        self.dt : float = dt
        # dt_arr - массив времени
        self.dt_arr = None
        
        return

        
    def step(self, args_pred_ar=(), args_upd_ar=()):
        '''
        Выполнить шаг фильтра, предсказание + обновление. Также обновляет знвчение dt, 
        если используется dt_arr.

        Args:
        args_pred_ar (arguments, optional)
            : аргументы функции predict используемых фильтров. 
        args_upd_ar  (arguments, optional)
            : аргументы функции update используемых фильтров.

        Returns:
        None

        '''

        if self.dt_arr is not None: 
            if self.n == 0: self.dt = self.dt_arr[self.n]
            else:           self.dt = self.dt_arr[self.n] - self.dt_arr[self.n - 1]
            
        if not isinstance(args_pred_ar, tuple):
            args_pred_ar = (args_pred_ar,)

        if not isinstance(args_upd_ar, tuple):
            args_upd_ar = (args_upd_ar,)

        for i in range(self.N_filters):
            self.filters[i].predict(*args_pred_ar)
            self.filters[i].update(*args_upd_ar)
             
            
        for i in range(self.filters):
            if self.filters[i].epsilon(self.filters[i].P, self.filters[i].H, self.filters[i].R) < self.eps_max:
                c = i
                break

        self.x = self.filters[c].x
        self.r = self.filters[c].r
        self.y = self.filters[c].y

        
        self._compute()
            
        if self.X is not None:
            self.X       = np.append(self.X,  [self.x], axis=0)
        else:
            self.X       = np.array(self.x, ndmin=2)

            
        
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
            
    def _compute(self):        
        '''
        Вычислить и сохранить различные парамаетры.

        Returns:
        None

        '''
        if self.res_arr is not None:
            if  np.size(self.x.shape) == 2:
                self.res_arr     = np.append(self.res_arr,     self.r,  axis=0)
            else:
                self.res_arr     = np.append(self.res_arr,    [self.r], axis=0)
        else:
            self.res_arr     = np.array(self.r, ndmin=2)
        
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
        S_det = np.linalg.det(S)
        likelihood = np.exp(-0.5 * self.y.T @ np.linalg.inv(S) @ self.y) / np.sqrt((2*np.pi)**self.dim_m * S_det)
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
        return self.y.T @ np.linalg.inv(S) @ self.y
    
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
    
    def plot_r(self, dt = 1, axis_x=0, axis_h=0, sigma=0, stds=1, loc=4, name=None, **kwargs):      
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
        plot_x = np.arange(0, len(plot_y)*dt, dt)

        fig, ax = plt.subplots()
        fig = plt.plot(plot_x, plot_y, color='blue', label='residual', **kwargs)
        if (sigma == 1):
            std = np.sqrt(self.P_posterior) * stds
            plot_y = np.array(self.P_posterior[:, axis_x, axis_x])
        
            fig = plt.plot(plot_x,  plot_y, color='red', label='sigma', **kwargs)
            fig = plt.plot(plot_x, -plot_y, color='red', **kwargs)
                        
        plt.legend(loc=loc)
        plt.grid()
        if name is None:
            ax.set_title("Невязка")
        else:
            ax.set_title("Невязка по " + name)

        return 
        
    
    def plot_l(self, dt=1, stds=1, loc=4, **kwargs):     
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
        plot_x = np.arange(0, len(plot_y)*dt, dt)
        
        fig, ax = plt.subplots()
        fig = plt.plot(plot_x, plot_y, color='blue', label='likelihood', **kwargs)

        plt.legend(loc=loc)
        plt.grid()
        ax.set_title("Функция правдоподобия")

        return    
    
    def plot_e(self, dt=1, stds=1, loc=4, **kwargs):
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
        plot_x = np.arange(0, len(plot_y)*dt, dt)
        
        fig, ax = plt.subplots()
        fig = plt.plot(plot_x, plot_y, color='blue', label='epsilon', **kwargs)

        plt.legend(loc=loc)
        plt.grid()
        ax.set_title("Квадрат невязки, нормированный на матрицу S")

        return 
    
    def plot_3d(self, dim, radar_pos=None):
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
        print(self.X)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lim_z = self.X[int(len(self.X[:, 0])/2)][dim]
        ax.set_xlim(self.X[0][0]/1000, self.X[-1][0]/1000); ax.set_ylim(self.X[0][2*dim]/1000, self.X[-1][2*dim]/1000); ax.set_zlim(0, lim_z/1000);
        ax.scatter(self.X[:, 0]/1000, self.X[:, 2*dim]/1000, self.X[:, dim]/1000, label='parametric curve', color='red')
        if radar_pos is not None:
            ax.scatter(radar_pos[0]/1000, radar_pos[2]/1000, radar_pos[1]/1000, label='parametric curve', color='orange')
        ax.set_xlabel('km')
        ax.set_ylabel('km')
        ax.set_zlabel('km')
        ax.set_title("Отфильтрованная траектория")
        
        return


    
    def plot_compare_res(self, new_data, axis_h=0, name=None, loc=4, save_name=None, plot=True, **kwargs):
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
        dim = 3
        filter_data = np.array([self.X[:, 0], self.X[:, dim], self.X[:, 2*dim]]).T
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