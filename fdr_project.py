# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:31:01 2024

@author: jayde
"""
import os
os.chdir(r'D:\subjects\PHD\2409\MATH5472\project')

from fdr_bayes import fdr_by
from fdr_emp import fdr_emp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import datetime
import scipy.stats as stats
#%%function
def lfdr_true(beta_hat,sigma_s,null_mean,null_sigma,null_pi,mix_portion,mix_mean,mix_sigma):
    b_null=norm.pdf(beta_hat,loc=null_mean,scale=np.sqrt(null_sigma**2+sigma_s**2))*null_pi
    #post betahat
    b_all=b_null
    for pi_i,mean_i,sigma_i in zip(mix_portion,mix_mean,mix_sigma):
        pi_adj=(1-null_pi)*pi_i
        b_all+=pi_adj*norm.pdf(beta_hat,loc=mean_i,scale=np.sqrt(sigma_s**2+sigma_i**2))
    return b_null/b_all

def post_beta_prob_true(b,beta_hat,sigma_s,null_mean,null_sigma,null_pi,mix_portion,mix_mean,mix_sigma):
    betahat_cond=norm.pdf(beta_hat,loc=b,scale=sigma_s)
    #prior beta
    b_prob=norm.pdf(b,loc=null_mean,scale=null_sigma)*null_pi
    for pi_i,mean_i,sigma_i in zip(mix_portion,mix_mean,mix_sigma):
        pi_adj=(1-null_pi)*pi_i
        b_prob+=pi_adj*norm.pdf(b,loc=mean_i,scale=sigma_i)
    
    #post betahat
    b_hat=norm.pdf(beta_hat,loc=null_mean,scale=np.sqrt(null_sigma**2+sigma_s**2))*null_pi
    for pi_i,mean_i,sigma_i in zip(mix_portion,mix_mean,mix_sigma):
        pi_adj=(1-null_pi)*pi_i
        b_hat+=pi_adj*norm.pdf(beta_hat,loc=mean_i,scale=np.sqrt(sigma_s**2+sigma_i**2))

    return betahat_cond*b_prob/b_hat

def lfsr_true(beta_hat,sigma_s,null_mean,null_sigma,null_pi,mix_portion,mix_mean,mix_sigma):
    
    lower_limit = -np.inf 
    upper_limit = np.inf
    boundary_value=1
    #positive
    pos_result, pos_error = quad(post_beta_prob_true, -1*boundary_value, upper_limit,args=(beta_hat,sigma_s,null_mean,null_sigma,null_pi,mix_portion,mix_mean,mix_sigma,))
    #negative
    neg_result, neg_error = quad(post_beta_prob_true, lower_limit,boundary_value,args=(beta_hat,sigma_s,null_mean,null_sigma,null_pi,mix_portion,mix_mean,mix_sigma,))
    
    #print(pos_result,pos_error,neg_result,neg_error)
    return min(pos_result,neg_result)

#%%prove for pi0 accurcay;lfdr,lfsr robust test
#null proportion accuracy test
pi0_ls=np.linspace(0.1,0.7,10)
n=10000
null_sigma=0.0001
null_mean=0
n_loc_cal=100
pi0_by=[]
pi0_emp=[]
figure_type={
    'spiky':{
        'weights':[0.4,0.2,0.2,0.2],
        'means':[0,0,0,0],
        'sigma_all':[0.25,0.5,0.1,0.2]
        },
    
    'near_normal':{
        'weights':[2/3,1/3],
        'means':[0,0],
        'sigma_all':[1,2]
        },    
    'flattop':{
        'weights':np.full(7,1/7),
        'means':[-1.5,-1,-0.5,0,0.5,1.0,1.5],
        'sigma_all':np.full(7,0.5)
        },
    
    'skew':{
        'weights':[1/4,1/4,1/3,1/6],
        'means':[-2,-1,0,1],
        'sigma_all':[2,1.5,1,1]
        },    
    'big_normal':{
        'weights':[1],
        'means':[0],
        'sigma_all':[4]
        },
    
    'bimodal':{
        'weights':[0.5,0.5],
        'means':[-2,2],
        'sigma_all':[1,1]
        }
    }

lfdr_by={}
lfsr_by={}
pi0_by={}
pi0_emp={}
lfdr_emp={}
lfdr_true_dic={}
lfsr_true_dic={}


for key_i,value_i in figure_type.items():
    figure_name=key_i
    
    lfdr_by[key_i]={}
    lfsr_by[key_i]={}
    pi0_by[key_i]=[]
    pi0_emp[key_i]=[]
    lfdr_emp[key_i]={}
    lfdr_true_dic[key_i]={}
    lfsr_true_dic[key_i]={}
    
    for pi0_i in pi0_ls:
        
        #generate samples
        null_size=int(n*pi0_i)
        non_null_size=n-null_size
        measurement_sigma=1
        
        null_set_mu=[]
        null_set_sigma=[]
        for _ in range(null_size):
            sample=np.random.normal(loc=null_mean, scale=null_sigma)
            sample_beta=np.random.normal(loc=sample,scale=measurement_sigma)
            null_set_mu.append(sample_beta)
            null_set_sigma.append(measurement_sigma)
        
        
        #sample alternative sets
        #define the parameters of the GMM
        weights = value_i['weights']  # Mixing coefficients
        means = value_i['means']  # Means of the components
        sigma_all = value_i['sigma_all']  # Standard deviations of the components
        #sample from the GMM
        alt_set_mu = []
        alt_set_sigma=[]
        for _ in range(non_null_size):
            #choose a component based on the mixing coefficients
            component = np.random.choice(len(weights), p=weights)
            #sample from the chosen Gaussian component
            sample = np.random.normal(loc=means[component], scale=sigma_all[component])
            
            sample_beta=np.random.normal(loc=sample,scale=measurement_sigma)
            alt_set_mu.append(sample_beta)
            alt_set_sigma.append(measurement_sigma)
            
        
        #combined data
        beta=np.array(null_set_mu+alt_set_mu)
        s=np.array(null_set_sigma+alt_set_sigma)
        
        #lfdr/lfsr sample
        index_s_null=np.random.choice(np.arange(0,null_size),size=int(n_loc_cal*pi0_i),replace=False)
        index_s_alt=np.random.choice(np.arange(null_size,n),size=n_loc_cal-int(n_loc_cal*pi0_i),replace=False)
        index_s=np.concatenate([index_s_null,index_s_alt])
        sample_beta_hat=list([beta[index_i],s[index_i]] for index_i in index_s)
        
        #true prob param
        null_mean=null_mean
        null_sigma=null_sigma
        null_pi=pi0_i
        mix_portion=weights
        mix_mean=means
        mix_sigma=sigma_all
        
        #fdr_bayes
        fdr_s=fdr_by(beta,s)
        fdr_s.fit()
        
        #print('fdr_by',datetime.datetime.now())
        fdr_s.lfsr(sample_beta_hat[0][0],sample_beta_hat[0][1])
        
        lfdr_by[key_i][pi0_i]=list([fdr_s.lfdr(beta_hat,sigma_s) for beta_hat,sigma_s in sample_beta_hat])
        lfsr_by[key_i][pi0_i]=list([fdr_s.lfsr(beta_hat,sigma_s) for beta_hat,sigma_s in sample_beta_hat])
        pi0_by[key_i].append(fdr_s.pi[0])
        
        lfdr_true_dic[key_i][pi0_i]=list([lfdr_true(beta_hat,sigma_s,null_mean,null_sigma,null_pi,mix_portion,mix_mean,mix_sigma) for beta_hat,sigma_s in sample_beta_hat])
        lfsr_true_dic[key_i][pi0_i]=list([lfsr_true(beta_hat,sigma_s,null_mean,null_sigma,null_pi,mix_portion,mix_mean,mix_sigma) for beta_hat,sigma_s in sample_beta_hat])
        
        #fdr emp
        fdr_s=fdr_emp(beta/s)
        fdr_s.fit()
        #print('fdr_emp',datetime.datetime.now())
        pi0_emp[key_i].append(fdr_s.pi0)
        lfdr_emp[key_i][pi0_i]=list([fdr_s.lfdr(beta_hat/sigma_s) for beta_hat,sigma_s in sample_beta_hat])
        
#plot to companre calculation accuracy for pi0
#create a figure with 4 subplots in one row
fig, axs = plt.subplots(1, 6, figsize=(20, 5), sharex=True, sharey=True)
plt_cnt=0
for key_i in pi0_by.keys():
    line1=axs[plt_cnt].scatter(pi0_ls,pi0_by[key_i],color='blue',label='LFDR Bayes')
    line2=axs[plt_cnt].scatter(pi0_ls,pi0_emp[key_i],color='green',label='LFDR Empirical') 
    line3,=axs[plt_cnt].plot(pi0_ls,pi0_ls,color='red',linestyle='-',label='Benchmark')
    axs[plt_cnt].set_title(key_i)
    
    plt_cnt+=1
# Set common labels
fig.text(0.5, 0, 'True Null Proportion', ha='center')
fig.text(0, 0.5, 'Estimated Null Proportion', va='center', rotation='vertical')
# Add a title for the entire figure
subtitle='Null Proportion Estimation Accuracy Test Across Different Sampling Distribution'
fig.suptitle(subtitle, fontsize=16,y=1.1)
#create a single legend for all subplots
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1),ncol=3)
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the title
plt.savefig(subtitle+'.png',bbox_inches='tight')
plt.show()

#plot to show lfdr of bayes across different distribution
#create a figure with 4 subplots in one row
fig, axs = plt.subplots(1, 6, figsize=(20, 5), sharex=True, sharey=True)
plt_cnt=0
for key_i,value_i in lfdr_by.items():
    true_min=2
    true_max=0
    for key_j,value_j in value_i.items():
        axs[plt_cnt].scatter(lfdr_true_dic[key_i][key_j],lfdr_by[key_i][key_j],color='blue')
        
        true_min_temp=min(lfdr_true_dic[key_i][key_j])
        true_max_temp=max(lfdr_true_dic[key_i][key_j])
        if true_min_temp< true_min:
            true_min=true_min_temp
        if true_max_temp> true_max:
            true_max=true_max_temp
    b_x=np.linspace(true_min,true_max,len(lfdr_true_dic[key_i][key_j]))
    b_y=b_x
    axs[plt_cnt].plot(b_x,b_y,color='red',linestyle='-',label='Benchmark')
    axs[plt_cnt].set_title(key_i)
    plt_cnt+=1
# Set common labels
fig.text(0.5, 0, 'True LFDR', ha='center')
fig.text(0, 0.5, 'Estimated LFDR', va='center', rotation='vertical')
# Add a title for the entire figure
subtitle='LFDR-True VS Estimated Across Disfferent Distribution'
fig.suptitle(subtitle, fontsize=16,y=1.1)
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the title
plt.savefig(subtitle+'.png',bbox_inches='tight')
plt.show()

#plot to show lfsr of bayes across different distribution
#create a figure with 4 subplots in one row
fig, axs = plt.subplots(1, 6, figsize=(20, 5), sharex=True, sharey=True)
plt_cnt=0
for key_i,value_i in lfsr_by.items():
    true_min=2
    true_max=0
    for key_j,value_j in value_i.items():
        axs[plt_cnt].scatter(lfsr_true_dic[key_i][key_j],lfsr_by[key_i][key_j],color='blue')
        
        true_min_temp=min(lfsr_true_dic[key_i][key_j])
        true_max_temp=max(lfsr_true_dic[key_i][key_j])
        if true_min_temp< true_min:
            true_min=true_min_temp
        if true_max_temp> true_max:
            true_max=true_max_temp
    b_x=np.linspace(true_min,true_max,len(lfsr_true_dic[key_i][key_j]))
    b_y=b_x
    axs[plt_cnt].plot(b_x,b_y,color='red',linestyle='-',label='Benchmark')
    axs[plt_cnt].set_title(key_i)
    plt_cnt+=1
# Set common labels
fig.text(0.5, 0, 'True LFSR', ha='center')
fig.text(0, 0.5, 'Estimated LFSR', va='center', rotation='vertical')
# Add a title for the entire figure
subtitle='LFSR-True VS Estimated Across Disfferent Distribution'
fig.suptitle(subtitle, fontsize=16,y=1.1)
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the title
plt.savefig(subtitle+'.png',bbox_inches='tight')
plt.show()

#plot to compare lfdr from the 2 distributions
#create a figure with 4 subplots in one row
fig, axs = plt.subplots(1, 6, figsize=(20, 5), sharey=True)
plt_cnt=0
for key_i,value_i in lfdr_by.items():
    key_j=pi0_ls[5]
    line1=axs[plt_cnt].scatter(lfdr_true_dic[key_i][key_j],lfdr_by[key_i][key_j],color='blue',label='LFDR Bayes')
    line2=axs[plt_cnt].scatter(lfdr_true_dic[key_i][key_j],lfdr_emp[key_i][key_j],color='green',label='LFDR Empirical')
    true_min=min(lfdr_true_dic[key_i][key_j])
    true_max=max(lfdr_true_dic[key_i][key_j])

    b_x=np.linspace(true_min,true_max,len(lfdr_true_dic[key_i][key_j]))
    b_y=b_x
    line3,=axs[plt_cnt].plot(b_x,b_y,color='red',linestyle='-',label='Benchmark')
    axs[plt_cnt].set_title(key_i)
    plt_cnt+=1
# Set common labels
fig.text(0.5, 0, 'True LFDR', ha='center')
fig.text(0, 0.5, 'Estimated LFDR', va='center', rotation='vertical')
#create a single legend for all subplots
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1),ncol=3)
# Add a title for the entire figure
subtitle='LFDR Accuracy Test Across Different Sampling Distribution'
fig.suptitle(subtitle, fontsize=16,y=1.1)
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the title
plt.savefig(subtitle+'.png',bbox_inches='tight')
plt.show()
#%% Different Measurement Precision
#generate samples
n=10000
pi0=0.7
null_sigma=0.0001
null_mean=0
good_ratio=0.5
null_size=int(n*pi0)
non_null_size=n-null_size
measurement_good_sigma=1
measurement_bad_sigma=10
n_loc_cal=1000

null_set_good_mu=[]
null_set_good_sigma=[]
null_set_bad_mu=[]
null_set_bad_sigma=[]
#good null generation
for _ in range(int(null_size*good_ratio)):
    sample=np.random.normal(loc=null_mean, scale=null_sigma)
    sample_beta=np.random.normal(loc=sample,scale=measurement_good_sigma)
    null_set_good_mu.append(sample_beta)
    null_set_good_sigma.append(measurement_good_sigma)

#bad null generation
for _ in range(null_size-int(null_size*good_ratio)):
    sample=np.random.normal(loc=null_mean, scale=null_sigma)
    sample_beta=np.random.normal(loc=sample,scale=measurement_bad_sigma)
    null_set_bad_mu.append(sample_beta)
    null_set_bad_sigma.append(measurement_bad_sigma)

#sample alternative sets
#define the parameters of the GMM
value_i=figure_type['near_normal']
weights = value_i['weights']  # Mixing coefficients
means = value_i['means']  # Means of the components
sigma_all = value_i['sigma_all']  # Standard deviations of the components
#sample from the GMM
alt_set_good_mu = []
alt_set_good_sigma=[]
alt_set_bad_mu = []
alt_set_bad_sigma=[]
#good alt
for _ in range(int(non_null_size*good_ratio)):
    #choose a component based on the mixing coefficients
    component = np.random.choice(len(weights), p=weights)
    #sample from the chosen Gaussian component
    sample = np.random.normal(loc=means[component], scale=sigma_all[component])
    
    sample_beta=np.random.normal(loc=sample,scale=measurement_good_sigma)
    alt_set_good_mu.append(sample_beta)
    alt_set_good_sigma.append(measurement_good_sigma)
#bad alt
for _ in range(null_size-int(non_null_size*good_ratio)):
    #choose a component based on the mixing coefficients
    component = np.random.choice(len(weights), p=weights)
    #sample from the chosen Gaussian component
    sample = np.random.normal(loc=means[component], scale=sigma_all[component])
    
    sample_beta=np.random.normal(loc=sample,scale=measurement_bad_sigma)
    alt_set_bad_mu.append(sample_beta)
    alt_set_bad_sigma.append(measurement_bad_sigma)    

#combined data
beta_c=np.array(null_set_good_mu+null_set_bad_mu+alt_set_good_mu+alt_set_bad_mu)
s_c=np.array(null_set_good_sigma+null_set_bad_sigma+alt_set_good_sigma+alt_set_bad_sigma)

#good data
beta_g=np.array(null_set_good_mu+alt_set_good_mu)
s_g=np.array(null_set_good_sigma+alt_set_good_sigma)

#lfdr/lfsr sample mixed good and bad
index_s_null_good=np.random.choice(np.arange(0,int(non_null_size*good_ratio)),size=int(n_loc_cal*pi0*good_ratio),replace=False)
index_s_alt_good=np.random.choice(np.arange(int(non_null_size*good_ratio)+null_size-int(non_null_size*good_ratio),2*int(non_null_size*good_ratio)+null_size-int(non_null_size*good_ratio)),size=int(n_loc_cal*(1-pi0)*good_ratio),replace=False)
index_s_null_bad=np.random.choice(np.arange(int(non_null_size*good_ratio),int(non_null_size*good_ratio)+null_size-int(non_null_size*good_ratio)),size=int(n_loc_cal*pi0*(1-good_ratio)),replace=False)
index_s_alt_bad=np.random.choice(np.arange(2*int(non_null_size*good_ratio)+null_size-int(non_null_size*good_ratio),n),size=int(n_loc_cal*(1-pi0)*(1-good_ratio)),replace=False)
index_s=np.concatenate([index_s_null_good,index_s_null_bad,index_s_alt_good,index_s_alt_bad])
sample_beta_hat=list([beta_c[index_i],s_c[index_i]] for index_i in index_s)

#good model
fdr_by_g=fdr_by(beta_g,s_g)
fdr_by_g.fit()
fdr_emp_g=fdr_emp(beta_g/s_g)
fdr_emp_g.fit()
#all model
fdr_by_c=fdr_by(beta_c,s_c)
fdr_by_c.fit()
fdr_emp_c=fdr_emp(beta_c/s_c)
fdr_emp_c.fit()

#compare good and combined model based on sample
lfdr_by_g=list([fdr_by_g.lfdr(sample_beta_hat[i_s][0],sample_beta_hat[i_s][1]) for i_s in range(n_loc_cal)])
lfdr_by_c=list([fdr_by_c.lfdr(sample_beta_hat[i_s][0],sample_beta_hat[i_s][1]) for i_s in range(n_loc_cal)])
lfsr_by_g=list([fdr_by_g.lfsr(sample_beta_hat[i_s][0],sample_beta_hat[i_s][1]) for i_s in range(n_loc_cal)])
print(datetime.datetime.now())
lfsr_by_c=list([fdr_by_c.lfsr(sample_beta_hat[i_s][0],sample_beta_hat[i_s][1]) for i_s in range(n_loc_cal)])
print(datetime.datetime.now())
lfdr_emp_g=list([fdr_emp_g.lfdr(sample_beta_hat[i_s][0]/sample_beta_hat[i_s][1]) for i_s in range(n_loc_cal)])
lfdr_emp_c=list([fdr_emp_c.lfdr(sample_beta_hat[i_s][0]/sample_beta_hat[i_s][1]) for i_s in range(n_loc_cal)])
print(datetime.datetime.now())


#create a figure with 3 subplots in one row
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
axs[0].scatter(lfdr_emp_g,lfdr_emp_c,color='blue')
b_x=np.linspace(min(lfdr_emp_g),max(lfdr_emp_c),n_loc_cal)
b_y=b_x
axs[0].plot(b_x,b_y,color='red',linestyle='-')
axs[0].set_title('LFDR Empirical')

axs[1].scatter(lfdr_by_g,lfdr_by_c,color='blue')
b_x=np.linspace(min(lfdr_by_g),max(lfdr_by_c),n_loc_cal)
b_y=b_x
axs[1].plot(b_x,b_y,color='red',linestyle='-')
axs[1].set_title('LFDR Bayes')

axs[2].scatter(lfsr_by_g,lfsr_by_c,color='blue')
b_x=np.linspace(min(lfsr_by_g),max(lfsr_by_c),n_loc_cal)
b_y=b_x
axs[2].plot(b_x,b_y,color='red',linestyle='-')
axs[2].set_title('LFSR Bayes')
# Add a title for the entire figure
subtitle='High Precision VS Combined'
fig.suptitle(subtitle, fontsize=16,y=1.1)
# Set common labels
fig.text(0.5, 0, 'Value from High Precision', ha='center')
fig.text(0, 0.5, 'Value from Combined Precision', va='center', rotation='vertical')
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the title
plt.savefig(subtitle+'.png',bbox_inches='tight')
plt.show()

#Good VS Bad lfsr-based on fdr bayes combined model
n_p_sample=100
p_value_ar=np.linspace(0.05,0.95,n_p_sample)
# Calculate the corresponding z-value
z_value=np.array([stats.norm.ppf(1-p_i/2) for p_i in p_value_ar])

beta_sample_g=z_value*measurement_good_sigma
beta_sample_b=z_value*measurement_bad_sigma

lfsr_b=list([fdr_by_c.lfsr(beta_sample_b[i_s],measurement_bad_sigma) for i_s in range(n_p_sample)])
lfsr_g=list([fdr_by_c.lfsr(beta_sample_g[i_s],measurement_good_sigma) for i_s in range(n_p_sample)])

#create the plot
plt.figure(figsize=(10, 6))
plt.plot(p_value_ar, lfsr_b, color='blue', label='sigma-10')
plt.plot(p_value_ar, lfsr_g, color='green', label='sigma-1')
# Add labels and title
plt.xlabel('P-value')
plt.ylabel('LFSR')
sub_title='Comparison of LFSR for Points from Different Measurement Precision'
plt.title(sub_title)
plt.legend()
plt.savefig(sub_title+'.png',bbox_inches='tight')
plt.show()

#True Postive VS False Positive (from combined model)
lfsr_fp=[]
lfsr_tp=[]
lfdr_fp=[]
lfdr_tp=[]
p_fp=[]
p_tp=[]
for p_i in p_value_ar:
    lfsr_fp_temp=0
    lfsr_tp_temp=0
    lfdr_fp_temp=0
    lfdr_tp_temp=0
    p_fp_temp=0
    p_tp_temp=0
    for i_s in range(n_loc_cal):
        #null set
        if i_s<int(n_loc_cal*pi0):
            #lfsr judgement
            if lfsr_by_c[i_s]<=p_i:
                lfsr_fp_temp+=1
            #lfdr judgement
            if lfdr_by_c[i_s]<=p_i:
                lfdr_fp_temp+=1
            #p judgement
            p_temp=2*(1-stats.norm.cdf(abs(sample_beta_hat[i_s][0]/sample_beta_hat[i_s][1])))
            if p_temp<=p_i:
                p_fp_temp+=1
        else:
            #alt set
            #lfsr judgement
            if lfsr_by_c[i_s]<=p_i:
                lfsr_tp_temp+=1
            #lfdr judgement
            if lfdr_by_c[i_s]<=p_i:
                lfdr_tp_temp+=1
            #p judgement
            p_temp=2*(1-stats.norm.cdf(abs(sample_beta_hat[i_s][0]/sample_beta_hat[i_s][1])))
            if p_temp<=p_i:
                p_tp_temp+=1
    lfsr_fp.append(lfsr_fp_temp)
    lfsr_tp.append(lfsr_tp_temp)
    lfdr_fp.append(lfdr_fp_temp)
    lfdr_tp.append(lfdr_tp_temp)
    p_fp.append(p_fp_temp)
    p_tp.append(p_tp_temp)
    
lfsr_fp=np.array(lfsr_fp)
lfsr_tp=np.array(lfsr_tp)
lfdr_fp=np.array(lfdr_fp)
lfdr_tp=np.array(lfdr_tp)
p_fp=np.array(p_fp)
p_tp=np.array(p_tp)
#create the plot
plt.figure(figsize=(10, 6))
sorted_indices_lfsr = np.argsort(lfsr_fp)
lfsr_fp_sorted = lfsr_fp[sorted_indices_lfsr]
lfsr_tp_sorted = lfsr_tp[sorted_indices_lfsr]
plt.plot(lfsr_fp_sorted,lfsr_tp_sorted, color='blue', label='LFSR')
sorted_indices_lfdr = np.argsort(lfdr_fp)
lfdr_fp_sorted = lfdr_fp[sorted_indices_lfdr]
lfdr_tp_sorted = lfdr_tp[sorted_indices_lfdr]
plt.plot(lfdr_fp_sorted,lfdr_tp_sorted, color='green', label='LFDR')
sorted_indices_p = np.argsort(p_fp)
p_fp_sorted = p_fp[sorted_indices_p]
p_tp_sorted = p_tp[sorted_indices_p]
plt.plot(p_fp_sorted, p_tp_sorted, color='red', label='P-value')
# Add labels and title
plt.xlabel('False Positives')
plt.ylabel('True Positives')
subtitle='Model Performance'
plt.title(subtitle)
plt.legend()
plt.savefig(subtitle+'.png',bbox_inches='tight')
plt.show()
#%%Penalty Effect
pi0=0.6
null_size=int(n*pi0)
non_null_size=n-null_size
measurement_sigma=1
value_i=figure_type['near_normal']
null_set_mu=[]
null_set_sigma=[]
for _ in range(null_size):
    sample=np.random.normal(loc=null_mean, scale=null_sigma)
    sample_beta=np.random.normal(loc=sample,scale=measurement_sigma)
    null_set_mu.append(sample_beta)
    null_set_sigma.append(measurement_sigma)


#sample alternative sets
#define the parameters of the GMM
weights = value_i['weights']  # Mixing coefficients
means = value_i['means']  # Means of the components
sigma_all = value_i['sigma_all']  # Standard deviations of the components
#sample from the GMM
alt_set_mu = []
alt_set_sigma=[]
for _ in range(non_null_size):
    #choose a component based on the mixing coefficients
    component = np.random.choice(len(weights), p=weights)
    #sample from the chosen Gaussian component
    sample = np.random.normal(loc=means[component], scale=sigma_all[component])
    
    sample_beta=np.random.normal(loc=sample,scale=measurement_sigma)
    alt_set_mu.append(sample_beta)
    alt_set_sigma.append(measurement_sigma)
#combined data
beta=np.array(null_set_mu+alt_set_mu)
s=np.array(null_set_sigma+alt_set_sigma)

#penalty_zero
fdr_by_0=fdr_by(beta,s)
fdr_by_0.fit()

#penalty_100
fdr_by_10=fdr_by(beta,s,lam0=1000)
fdr_by_10.fit()

print('null proportion from no penalty:',fdr_by_0.pi[0])
print('null proportion from lambda of 1000:',fdr_by_10.pi[0])

