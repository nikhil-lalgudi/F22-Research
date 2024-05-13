#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import celluloid
from celluloid import Camera
from scipy.special import expit
from matplotlib import gridspec

class LogisticRegression(object):
    def __init__(self,x,y, lr=0.01):
        self.lr=lr
        n=x.shape[1]
        self.w=np.ones((1,n))*(0) 
        self.b=0.5 
        
    def predict(self,x): 
        z=x@self.w.T  + self.b  
        p=expit(z) 
        return p 

    def cost(self, x,y): 
        z=x@self.w.T  + self.b
        p=expit(z)
        return - np.mean(y*np.log(p) + (1-y)*np.log(1-p)) 
    
    def step(self,x,y):
        z=x@self.w.T  + self.b
        p=expit(z)
    
        # Partial derivatives:       
        dw= np.mean((p - y)*x, axis=0)    
        db = np.mean(p-y)               
        self.w = self.w - dw*self.lr 
        self.b= self.b- db*self.lr  
        
    def fit(self,x,y,numberOfEpochs=100000):
        self.AllWeights=np.zeros((numberOfEpochs, x.shape[1]))
        self.AllBiases=np.zeros((numberOfEpochs, x.shape[1]))
        self.AllCosts=np.zeros((numberOfEpochs, x.shape[1]))
        self.All_cl= np.zeros((numberOfEpochs,len(x))) 
        
        for step in range(numberOfEpochs):
            self.AllWeights[step]=self.w  
            self.AllBiases[step]=self.b
            self.AllCosts[step]=self.cost(x,y)
            self.All_cl[step]=(self.predict(x)).T.flatten() 
            self.step(x,y) 


# In[2]:


x = pd.read_csv('data4_xvalue.csv', sep=',', header=None)
y = pd.read_csv('data4_yvalue.csv', sep=',', header=None)

epochs_=100000 
model=LogisticRegression(x.values,y.values, lr=0.001)  
model.fit(x.values,y.values, numberOfEpochs=epochs_)


print("-------- Multiple logistic regression model:")
print("Final weights: "+ str(model.w))
print("Final bias: "+ str(model.b))
print("Final costs: " + str(model.cost(x.values,y.values)))


class LogisticRegression_fixed_b(object):  
    def __init__(self,x,y,b,lr=0.01):
        self.lr=lr
        n=x.shape[1]
        self.w=np.array([[-0.1,-0.1]])
        self.b=np.array([[b]]) 
        
    def predict(self,x):
        p=expit( x @self.w.T + self.b)
        return p        
            
    def cost(self,x,y):     
        p=expit(x @ self.w.T + self.b)
        return - np.mean(y*np.log(p) + (1-y)*np.log(1-p))

    def step(self, x,y):
        p=expit(x @ self.w.T + self.b)
        e = p - y        
        dw= np.mean(e*x, axis=0)    
        self.w = self.w - dw*self.lr 
        
    def fit(self, x,y, numberOfEpochs=1000000):
        self.AllWeights= np.zeros((numberOfEpochs, x.shape[1]))
        self.AllBiases= np.zeros(numberOfEpochs)
        self.AllCosts= np.zeros(numberOfEpochs)
        self.All_cl= np.zeros((numberOfEpochs,len(x))) 
        for step in range(numberOfEpochs):
            self.AllWeights[step]=self.w  
            self.AllCosts[step]=self.cost(x,y)
            self.All_cl[step]=(self.predict(x)).T.flatten() 
            self.step(x,y)

b_fixed= float(model.b) 
model=LogisticRegression_fixed_b(x.values,y.values,b_fixed, lr=0.001) 
model.fit(x.values,y.values, numberOfEpochs=epochs_)
    
w0=model.AllWeights.T[0]
w1=model.AllWeights.T[1]
c=model.AllCosts
cl=model.All_cl

print("-------- Multiple logistic regression model (with fixed y-intercept): ")
print("Final weights: "+ str(model.w))
print("Final bias: "+ str(model.b))
print("Final costs: " + str(model.cost(x.values,y.values)))


# In[14]:


def pred_3d_curve(X,w,b):   
        p=expit(X @ w.T + b)
        return p
    
n0s = np.linspace(0.5, 2, 200) 
n1s = np.linspace(250, -250, 200)
N1, N2 = np.meshgrid(n0s, n1s) 

def CrossEntropy_cost(x,y,w,b): 
        p=expit(x @ w.T + b)
        return - np.mean(y*np.log(p) + (1-y)*np.log(1-p))
    
m0s = np.linspace(-0.12, 0.5, 35) 
m1s = np.linspace(-0.135, 0.6, 35)
M1, M2 = np.meshgrid(m0s, m1s) 
zs_1 = np.array([CrossEntropy_cost(x.values,y.values,       
                       np.array([[wp0,wp1]]), np.array([[b_fixed]]))
               for wp0, wp1 in zip(np.ravel(M1), np.ravel(M2))])
Z_1 = zs_1.reshape(M1.shape) 

# Create plot:
fig = plt.figure(figsize=(8,10)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[1.3, 1]) 


label_font_size = 25 
tick_label_size= 17 
ax0=fig.add_subplot(gs[0], projection="3d")
ax0.set_title("Logistic regression curve (3D)", fontsize=20) 
ax0.view_init(elev=38., azim=-25)
ax0.set_xlabel(r'$Scaled AdjGoals$', fontsize=label_font_size, labelpad=8)
ax0.set_ylabel(r'$Scaled AdjELO$', fontsize=label_font_size, labelpad=7)
ax0.set_zlabel("Probability", fontsize=label_font_size, labelpad=6)
ax0.tick_params(axis='both', which='major', labelsize=tick_label_size)
ax0.tick_params(axis='x', pad=3, which='major', labelsize=tick_label_size)
ax0.tick_params(axis='y', pad=-2, which='major', labelsize=tick_label_size)
ax1=fig.add_subplot(gs[1], projection="3d")
ax1.view_init(elev=38., azim=-25)
ax1.view_init(elev=38., azim=140)  
ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
ax1.tick_params(axis='x', pad=3, which='major', labelsize=tick_label_size)
ax1.tick_params(axis='y', pad=-2, which='major', labelsize=tick_label_size)
ax1.set_xlabel(r'$w_0$', fontsize=label_font_size, labelpad=14)
ax1.set_ylabel(r'$w_1$', fontsize=label_font_size, labelpad=14)
ax1.set_zlabel("costs", fontsize=label_font_size, labelpad=3)
ax1.set_xticks([0.5, 0.3,0.1,-0.1]) 
ax1.set_xticklabels(["0.5", "0.3","0.1","-0.1"], fontsize=tick_label_size)
ax1.set_yticks([0.6,0.4,0.2,0]) 
ax1.set_yticklabels(["0.6","0.4","0.2","0"], fontsize=tick_label_size)
ax1.set_zticks([0.6,0.7,0.8,0.9,1.0]) 
ax1.set_zticklabels(["0.6","0.7","0.8","0.9","1.0"], fontsize=tick_label_size)


a4=np.arange(99999,100000,82).tolist()
points_=a4

camera = Camera(fig) 
for i in points_:

    w=np.array([[w0[i],w1[i]]])
    zs_0 = np.array([pred_3d_curve(np.array([[wp0,wp1]]),w, np.array([[b_fixed]]))  
               for wp0, wp1 in zip(np.ravel(N1), np.ravel(N2))])
    Z_0 = zs_0.reshape(N1.shape) 
    ax0.plot_surface(N1, N2, Z_0, rstride=1, cstride=1,
                     alpha=0.4,cmap=cm.coolwarm,
                     antialiased=False)
    
    ax1.plot_surface(M1, M2, Z_1, rstride=1, cstride=1,
                     alpha=0.73,cmap=cm.coolwarm)
    
    ax0.legend([f'costs: {np.round(c[i],3)}'], loc=(0, 0.8), 
               fontsize=17)
    ax1.legend([f'epochs: {i}'], loc=(0, 0.8),
               fontsize=17)
    
    plt.tight_layout()
    camera.snap() 
    
animation = camera.animate(interval = 130, 
                          repeat = False, repeat_delay = 0) 
animation.save('LogReg_2.gif', writer = 'imagemagick') 


# In[ ]:


print(model.predict([0.42,35.62500]))

