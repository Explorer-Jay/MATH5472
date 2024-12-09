# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:15:14 2024

@author: jayde
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import datetime

class fdr_by:
    #the input of beta and s should be of array format
    def __init__(self,beta,s,lam0=False):
        self.beta=beta
        self.s=s
        self.lam0=lam0
        self.pi_0=0.6
        self.error_sum=0.01
        self.pi0_sigma=0.0001
    #calibration (fit)
    def fit(self):
        
        m=np.sqrt(2)
        ##sigma grid
        sigma_min=min(self.s)/10
        sigma_max=2*np.sqrt(max(self.beta**2-self.s**2))
        self.sigma_grid=np.array([self.pi0_sigma]+self.geometric_seq(sigma_min,sigma_max,m))
        self.k=len(self.sigma_grid)
        #pi estimation-EM
        self.pi=self.pi_em(self.sigma_grid)
        
    def geometric_seq(self,sigma_min,sigma_max,m):
        m_seq=[]
        current=sigma_min
        while current<sigma_max:
            m_seq.append(current)
            current*=m
        return m_seq
    def pi_em(self,sigma_grid):
        j=len(self.beta)
        #initialization
        k=len(sigma_grid)
        error_cur=10
        
        if self.lam0:
            lam_ls=[self.lam0]+[1]*(self.k-1)
        else:
            lam_ls=np.ones(k)
        pi_avg=(1-self.pi_0)/(k-1)
        pi_cur=np.array([1-pi_avg*(k-1)]+[pi_avg]*(k-1))
        
        iter_n=0
        self.error_ls=[]
        while error_cur>=self.error_sum and iter_n<=1000:
            #E-step-assume normal distribution
            #calculate l_k_j
            l_kj=np.zeros((k,j))
            for i_k in range(k):
                for i_j in range(j):
                    l_kj[i_k,i_j]=norm.pdf(self.beta[i_j],loc=0,scale=np.sqrt(self.s[i_j]**2+self.sigma_grid[i_k]**2))
            #calculate w_k_j
            w_kj=np.zeros((k,j))
            for i_k in range(k):
                for i_j in range(j):
                    w_kj[i_k,i_j]=pi_cur[i_k]*l_kj[i_k,i_j]/np.sum(pi_cur*l_kj[:,i_j])
            #calculate nk
            nk=np.array([np.sum(w_kj[k_i,:])+lam_i-1 for k_i,lam_i in zip(range(k),lam_ls)])
            #M-step
            pi_new=nk/np.sum(nk)
            
            error_cur=np.sum(abs(pi_new-pi_cur))
            pi_cur=pi_new
            
            iter_n+=1
            #print(error_cur)
            self.error_ls.append(error_cur)
        return pi_cur
    def lfdr(self,beta_hat,sigma_s):
        b_null=norm.pdf(beta_hat,loc=0,scale=np.sqrt(self.pi0_sigma**2+sigma_s**2))
        pi_0=self.pi[0]
        b_all=self.post_betahat_prob(beta_hat,sigma_s)
        #print(b_null,pi_0,b_all)
        return b_null*pi_0/b_all
    def post_betahat_prob(self,beta_hat,sigma_s):
        lb_partial=np.array([norm.pdf(beta_hat,loc=0,scale=np.sqrt(sigma_i**2+sigma_s**2)) for sigma_i in self.sigma_grid])
        return np.sum(self.pi*lb_partial)
    def post_beta_prob(self,b,beta_hat,sigma_s):
        betahat_cond=norm.pdf(beta_hat,loc=b,scale=sigma_s)
        
        beta_partial=np.array([norm.pdf(b,loc=0,scale=sigma_i) for sigma_i in self.sigma_grid])
        beta_prob=np.sum(self.pi*beta_partial)
        
        betahat_post=self.post_betahat_prob(beta_hat,sigma_s)

        return betahat_cond*beta_prob/betahat_post
    def lfsr(self,beta_hat,sigma_s):
        
        lower_limit = -np.inf 
        upper_limit = np.inf
        boundary_value=1
        #positive
        pos_result, pos_error = quad(self.post_beta_prob, -1*boundary_value, upper_limit,args=(beta_hat,sigma_s,))
        #negative
        neg_result, neg_error = quad(self.post_beta_prob, lower_limit,boundary_value,args=(beta_hat,sigma_s,))
        #print(pos_result,pos_error,neg_result,neg_error)
        return min(pos_result,neg_result)
        

    

    


    