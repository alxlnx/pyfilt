# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:28:20 2023

@author: y.ivashkevich
"""

from FilterLib.Kalman import Kalman
import numpy as np

class Fading(Kalman):
    '''
        Увядающий фильтр.
    '''
    def predict(self, alpha):
        '''
        Выполнить этап предсказания
        
        $$ X_prior = F*X_post + B*u $$
        $$ P_prior = alpha**2*F*P_post*F^T + Q $$

        Если это первый шаг фильтра, также вызывает функцию _first_step

        Args:
        alpha (float)
            : Коэффициент увядания.

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
            self.first_step()

        # X_prior = F*X_post + B*u
        self.x = self.F @ self.x + self.B @ self.u
        
        # P_prior = alpha**2*F*P_post*F^T + Q
        self.P = self.alpha**2 * self.F @ self.P @ self.F.T + self.Q

        self.prior   = np.append(self.prior,   [self.x], axis=0)
        self.P_prior = np.append(self.P_prior, [self.P], axis=0)
        
        self.predict_stage = False
        return
    