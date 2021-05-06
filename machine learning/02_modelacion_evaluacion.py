import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

mamoDataX = pd.read_csv('../csv/prueba.csv')
mamoDataY = mamoDataX['Tissue'] # this is the feature objective
mamoDataX = mamoDataX.drop(['Tissue'],axis=1) # this are the fetures neccesaries to compute the festure objective

# model and prediction

X_train, X_test, Y_train, Y_test = train_test_split(mamoDataX,mamoDataY, random_state = 2) # to split mamoData(X - Y) in (X - Y) trian /test

print(len(Y_test))
model = Lasso() # assignating model

model.fit(X_train,Y_train) # training the model
# print(X_test)
PREDICTED = model.predict(X_test) # to predict the objective data accurate to this data

'''
    Very bad results (neither mamoDataX (without zeros) neither mamoData (Tissue = 0 | 1) 
    neither mamoData (Tissue = 0 - 1). generate a good score) for that we are gonna aplly featuring engineering and
    evaluate other models 
'''

plt.subplot(2,2,1)
plt.hist([PREDICTED,Y_test]) # to know the relation the predict data with the Y_test data 

print(model.score(X_test,Y_test)) # score of the model (very bad!!)

RESIDUALS = Y_test - PREDICTED # to know the rate of the error
plt.subplot(2,2,2)
plt.scatter(Y_test,RESIDUALS) 

plt.subplot(2,2,3)
plt.hist(RESIDUALS, bins=100, normed = 1, histtype = 'step')

plt.show()

# map to correlation
# sb.heatmap(mamoDataX.corr()) # there are much correlation between the data that is bad 
