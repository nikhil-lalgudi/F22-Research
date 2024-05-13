#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.special import expit

# Load data
x = pd.read_csv('data5_xvalue.csv', sep=',', header=None)
y = pd.read_csv('data5_yvalue.csv', sep=',', header=None)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# SVM model
svm_model = svm.SVC()
svm_model.fit(x_train, y_train.values.ravel())
svm_predictions = svm_model.predict(x_test)

# Fixed intercept logistic regression model
class LogisticRegression(object):
    def __init__(self, x, y, lr=0.01):
        self.lr = lr
        n = x.shape[1]  # determine the number of independent variables
        self.w = np.ones((1, n)) * 0  # initialize weight matrix and set weights to zero
        self.b = 0.5  # set starting value for b to 0.5

    def predict(self, x):
        z = x @ self.w.T + self.b  # @: matrix multiplication
        p = expit(z)  # logistic sigmoid function
        return p

    def cost(self, x, y):
        z = x @ self.w.T + self.b
        p = expit(z)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))  # Cross-entropy cost function

    def step(self, x, y):
        z = x @ self.w.T + self.b
        p = expit(z)

        # Partial derivatives:
        dw = np.mean((p - y) * x, axis=0)  # dJ/dw
        db = np.mean(p - y)  # dJ/db
        self.w = self.w - dw * self.lr  # update w
        self.b = self.b - db * self.lr  # update b

    def fit(self, x, y, numberOfEpochs=100000):
        self.AllWeights = np.zeros((numberOfEpochs, x.shape[1]))
        self.AllBiases = np.zeros((numberOfEpochs, x.shape[1]))
        self.AllCosts = np.zeros((numberOfEpochs, x.shape[1]))
        self.All_cl = np.zeros((numberOfEpochs, len(x)))  # cl: predicted y-values for connection lines

        for step in range(numberOfEpochs):
            self.AllWeights[step] = self.w
            self.AllBiases[step] = self.b
            self.AllCosts[step] = self.cost(x, y)
            self.All_cl[step] = (self.predict(x)).T.flatten()
            self.step(x, y)

# Train multiple logistic regression model:
epochs_ = 100000  # number of epochs for training
model = LogisticRegression(x.values, y.values, lr=0.001)
model.fit(x.values, y.values, numberOfEpochs=epochs_)

# Return final model parameters and costs:
print("-------- Multiple logistic regression model:")
print("Final weights: " + str(model.w))
print("Final bias: " + str(model.b))
print("Final costs: " + str(model.cost(x.values, y.values)))

# Define new logistic reg. model with fixed-intercept:
class LogisticRegression_fixed_b(object):
    def __init__(self, x, y, b, lr=0.01):
        self.lr = lr
        n = x.shape[1]
        self.w = np.array([[-0.1, -0.1]])  # set starting values for weights to -0.1
        self.b = np.array([[b]])  # fixed value for b

    def predict(self, x):
        p = expit(x @ self.w.T + self.b)
        return p

    def cost(self, x, y):
        p = expit(x @ self.w.T + self.b)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def step(self, x, y):
        p = expit(x @ self.w.T + self.b)
        e = p - y
        dw = np.mean(e * x, axis=0)
        self.w = self.w - dw * self.lr

    def fit(self, x, y, numberOfEpochs=1000000):
        self.AllWeights = np.zeros((numberOfEpochs, x.shape[1]))
        self.AllBiases = np.zeros(numberOfEpochs)
        self.AllCosts = np.zeros(numberOfEpochs)
        self.All_cl = np.zeros((numberOfEpochs, len(x)))
        for step in range(numberOfEpochs):
            self.AllWeights[step] = self.w
            self.AllCosts[step] = self.cost(x, y)
            self.All_cl[step] = (self.predict(x)).T.flatten()
            self.step(x, y)

# Set y-intercept value and train fixed-intercept logistic regression model:
b_fixed = float(model.b)
model_fixed = LogisticRegression_fixed_b(x_train.values, y_train.values, b_fixed, lr=0.001)
model_fixed.fit(x_train.values, y_train.values, numberOfEpochs=epochs_)

# Stored parameter values of fixed-intercept logistic regression model:
w0_fixed = model_fixed.AllWeights.T[0]
w1_fixed = model_fixed.AllWeights.T[1]
c_fixed = model_fixed.AllCosts
cl_fixed = model_fixed.All_cl

# Evaluate models
svm_tp = confusion_matrix(y_test, svm_predictions)[1, 1]
svm_tn = confusion_matrix(y_test, svm_predictions)[0, 0]
svm_fp = confusion_matrix(y_test, svm_predictions)[0, 1]
svm_fn = confusion_matrix(y_test, svm_predictions)[1, 0]

fixed_lr_predictions = model_fixed.predict(x_test.values)
fixed_lr_tp = confusion_matrix(y_test, fixed_lr_predictions.round())[1, 1]
fixed_lr_tn = confusion_matrix(y_test, fixed_lr_predictions.round())[0, 0]
fixed_lr_fp = confusion_matrix(y_test, fixed_lr_predictions.round())[0, 1]
fixed_lr_fn = confusion_matrix(y_test, fixed_lr_predictions.round())[1, 0]

# Print the results
print("SVM True Positives:", svm_tp)
print("SVM True Negatives:", svm_tn)
print("SVM False Positives:", svm_fp)
print("SVM False Negatives:", svm_fn)

print("Fixed Intercept LR True Positives:", fixed_lr_tp)
print("Fixed Intercept LR True Negatives:", fixed_lr_tn)
print("Fixed Intercept LR False Positives:", fixed_lr_fp)
print("Fixed Intercept LR False Negatives:", fixed_lr_fn)

