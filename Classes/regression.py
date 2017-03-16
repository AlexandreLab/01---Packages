# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import calendar


#inspire from http://www.eeperformance.org/uploads/8/6/5/0/8650231/ashrae_-_inverse_modeling_toolkit_-_numerical_algorithms.pdf


class Regression:
    
    def __init__(self):
        print("Regression")
        self.beta1=0
        self.beta2=0
        self.beta3=0
        self.data = pd.DataFrame()
        self.param = [0,0,0]
        #LEAST-SQUARES REGRESSION ALGORITHM     
        
    def _normalized_by_mean(self, df):
        X_normalizer=df["Driver"].mean()
        Y_normalizer= df["Energy"].mean()   
        for col in df.columns:
            df.loc[: ,col]=df.loc[:, col]/df[col].mean()
        
        return X_normalizer, Y_normalizer
        
    
    def _model(self, df, normalize=True, verbose_eval = True):
        df.columns=["Energy", "Driver"]
        self.data = df.copy()
        
        if normalize:
            X_normalizer, Y_normalizer = self._normalized_by_mean(df)
        else:
            X_normalizer=1
            Y_normalizer=1
        
        X_min = df["Driver"].min()
        X_max = df["Driver"].max()
        X_mean = df["Driver"].mean()
        Y_mean = df["Energy"].mean()

        beta2 = self._get_beta2(df)
        beta1 = (Y_mean*X_mean - self.beta2 * X_mean)
        beta3=X_min

        #First iteration
        interval = (X_max - X_min)/10
        
        X_min_new= self._find_parameters(df, X_min, beta1, beta2, beta3, interval, 10, verbose_eval)
        
        sliced_df= df.loc[df["Driver"]<self.beta3, :].copy()
        self.beta2 = self._get_beta2(sliced_df)
        
        #Second iteration
        self._find_parameters(df, X_min_new, self.beta1+abs(self.beta2*interval*2), self.beta2, X_min_new, interval/10, 20, verbose_eval) 
        
        sliced_df= df.loc[df["Driver"]<self.beta3, :].copy()
        self.beta2 = self._get_beta2(sliced_df)
        
        #End results
        self.beta2 =self.beta2*Y_normalizer/X_normalizer
        self.beta1 = self.beta1*Y_normalizer
        self.beta3=self.beta3*X_normalizer
        self.param = [self.beta1, self.beta2, self.beta3]
        
        
        

    def _find_parameters(self, df, X_min, beta1, beta2, beta3, interval, range_size, verbose_eval):
        
        df["Predicted"]=[beta1+beta2*(x-beta3) if x<beta3 else beta1 for x in df["Driver"].values]
        X_min_new=X_min
        min_RMSE=self._get_RMSE(df)
        if verbose_eval:
            print("STARTING POINT" , "RMSE:", min_RMSE, "equation: Y=", beta1, "+", beta2,"*X", "(T-", beta3, ")")

        #beta1=beta1+abs(beta2*interval)
        
        for step in range(0, range_size, 1):
            
            beta3=X_min+step*interval
            
            beta1=beta1-abs(beta2*interval)
            if beta1< 0: beta1=0
            
            df["Predicted"]=[beta1+beta2*(x-beta3) if x<beta3 else beta1 for x in df["Driver"].values]
            RMSE=self._get_RMSE(df)
            if verbose_eval:
                print("step: ", step, "RMSE:", RMSE, "equation: Y=", beta1, "+", beta2,"*X", "(T-", beta3, ")")

            if RMSE<min_RMSE:
                
                #print(step, beta1, beta3, "RMSE",RMSE)
                self.beta1=beta1
                self.beta2=beta2
                self.beta3=beta3
                min_RMSE=RMSE
                X_min_new = beta3-interval*2
                
        return X_min_new
        
    def _get_beta2(self, df):
        X_mean = df["Driver"].mean()
        Y_mean = df["Energy"].mean()
        Sxx= (df["Driver"]-X_mean).dot((df["Driver"]-X_mean))
        Sxy =(df["Driver"]-X_mean).dot((df["Energy"]-Y_mean))
        return Sxy/Sxx

    def _get_RMSE(self, df):
        return (((df.Predicted - df.Energy) ** 2).sum()/(df.shape[0]-2)) ** .5

    def _print_steps(self,df, step, beta1, beta2, beta3):
        print("step: ", step, "RMSE:", self._get_RMSE(df), "Change point: ", beta1, beta2, beta3)
    
        
    def _plot(self):
        x_data = "Driver"
        y_data = "Energy"
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.regplot(x=x_data, y=y_data, data=self.data, fit_reg=False, ax=ax)
        std=self.data["Driver"].std()/5
        x = np.linspace(self.data["Driver"].min()-std, self.data["Driver"].max()+std, 10)
        y=[(self.beta1+self.beta2*(val-self.beta3)) if val<self.beta3 else self.beta1 for val in x]
        plt.plot(x, y, color='red')
        ax.set_xlabel("Driver")
        ax.set_ylabel("Energy")
        print("equation: Y=", "{:f}".format(self.beta1)+ "{:+f}".format(self.beta2)+"*X*(T-"+ "{:f}".format(self.beta3), ")")



