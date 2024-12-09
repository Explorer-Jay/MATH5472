# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:09:44 2024

@author: jayde
"""
import numpy as np
import statsmodels.api as sm 
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
import datetime

class fdr_emp():
    def __init__(self,z):
        self.z=z
        self.k=100
        self.degree=7
        self.zero_upper=0.001
        self.zero_lower=-0.001
        self.zero_n=100
    def fit(self):
        #calibrate fz
        #chunk data
        chunks,self.interval_width=self.chunk_data(self.z,self.k)
        
        v=np.array([len(c_i) for c_i in chunks if len(c_i)>0])
        z_mid=np.array([np.median(c_i) for c_i in chunks if len(c_i)>0])
        
        error_cnt=0
        protection=True
        while error_cnt<=10 and protection:
            try:
                poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                z_poly=poly.fit_transform(z_mid.reshape(-1,1))
                z_poly=sm.add_constant(z_poly)
                self.check_1=v
                self.check_2=z_poly
                self.fz=sm.GLM(v, z_poly, family=sm.families.Poisson()).fit()
                protection=False
            except Exception as e:
                self.degree+=1
                error_cnt+=1
                
                
        
        #calibrate f0
        f0z=np.linspace(self.zero_lower, self.zero_upper,self.zero_n)
        f0=self.fz_predict(f0z)
        c2,c1,c0 = np.polyfit(f0z, np.log(f0), 2)
        sigma_sq=max(-1/(2*c2),0.0000001)
        
        self.check_1=f0z
        self.check_2=f0
        #print(c2,c1,c0)
        
        self.u0=c1*sigma_sq
        self.pi0=min(np.exp(c0+0.5*(c1+np.log(2*np.pi*sigma_sq))),0.99)
        self.sigma0=np.sqrt(sigma_sq)
        
    #the input of x should be of array format
    def fz_predict(self,x):
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        x=x.reshape(-1,1)
        x_poly=poly.fit_transform(x)
        if len(x)==1:
            x_poly=np.insert(x_poly, 0,1, axis=1)
        else:
            x_poly=sm.add_constant(x_poly)
        return self.fz.predict(x_poly)/(len(self.z)*self.interval_width)
    def chunk_data(self,data, k):
        #find the maximum and minimum values
        min_val = min(data)
        max_val = max(data)
        
        #calculate the interval width
        interval_width = (max_val - min_val) / k
        
        #create chunks
        chunks = [[] for _ in range(k)]
        for value in data:
            # Determine the interval index
            index = min(int((value - min_val) / interval_width), k - 1)
            chunks[index].append(value)
        return chunks,interval_width
    def lfdr(self,x):
        f0=norm.pdf(x,loc=self.u0,scale=self.sigma0)
        fz=self.fz_predict(np.array(x))
        lfdr_v=self.pi0*f0/fz
        return min(lfdr_v[0],1)