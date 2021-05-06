import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures # To scaler and create features
from sklearn.pipeline import make_pipeline # to do the things faster

''' In this case we are gonna scaler and create new features to know if is beneficial to our model '''

mamoDataX = pd.read_csv("../csv/newMamoData.csv")
mamoDataY = mamoDataX['Tissue']
mamoDataX = mamoDataX.drop('Tissue', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(mamoDataX,mamoDataY, random_state = 2)

''' Now we are gonna scale the data features to try to improve the score '''

SCALER = StandardScaler()  
SCALER.fit(X_train)

# print(X_train)
# print(SCALER.transform(X_train)) # to scale the data centering in 0

X_train_scaled, X_test_scaled = (SCALER.transform(X_train),SCALER.transform(X_test)) # tranform data feature

model = Lasso()
model_scaled = Lasso()

model.fit(X_train,Y_train)
model_scaled.fit(X_train_scaled,Y_train)

print("score to normal model: {}" .format(model.score(X_test,Y_test)))
print("score to scaler data: {}" .format(model_scaled.score(X_test_scaled,Y_test))) # a worse score lol

''' Now we are gonna create more feature to try to improve the score '''

TRANSFORMER = PolynomialFeatures(2) # that number is according a formula
print("Dimensions normal DataFrame: {}" .format(mamoDataX.shape))
print("Dimensions tranformer DataFrame: {}" .format(TRANSFORMER.fit_transform(mamoDataX).shape))

model_poly = make_pipeline(PolynomialFeatures(2), Lasso())
# model_poly = model_poly.drop(['0'], axis = 1)
model_poly.fit(X_train,Y_train)


print("score to polynomial fetures model: {}" .format(model_poly.score(X_test,Y_test))) # a good score


''' How we watched polynomial features generate good results for that we are gonna create that csv file '''

prueba = TRANSFORMER.fit_transform(mamoDataX)
model_poly = TRANSFORMER.fit_transform(mamoDataX)
model_poly = pd.DataFrame(model_poly[:,1:])
model_poly.columns = list(mamoDataX.columns) +['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15']
model_poly.index = mamoDataX.index
model_poly = pd.concat([model_poly,mamoDataY], axis = 1)
model_poly.to_csv('../csv/prueba.csv', index = False)