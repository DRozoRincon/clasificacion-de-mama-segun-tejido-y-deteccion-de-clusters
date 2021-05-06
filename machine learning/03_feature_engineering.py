import numpy as np 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, mutual_info_regression

''' We are gonna try to improve the score of the model, studying data features '''

mamoDataX = pd.read_csv('../csv/mamoData3.csv')
mamoDataY = mamoDataX['Tissue']
mamoDataX = mamoDataX.drop(['Tissue'], axis = 1)

model = Lasso()

X_train, X_test, Y_train, Y_test = train_test_split(mamoDataX,mamoDataY, random_state = 2)

model.fit(X_train,Y_train)

''' Now we are gonna study what are the more importante feature to our model about its coefficient'''

COEF = model.coef_ # between bigger th coef the feture is better aport more to the prediction
# print(var)
plt.subplot(1,2,1)
plt.plot(COEF) # how we can watch the best are [P1,P3,Std3,N3] ...
plt.xticks(np.arange(9),list(mamoDataX.columns)) 
plt.title("Coefficient")

''' now we are goona study how is the correlaction beetween our feture objective and the others '''

mamoData = pd.concat([mamoDataX,mamoDataY], axis = 1)
# sb.pairplot(mamoData) # to know the realtion between features

''' we are gonna analyze what are the best feature, with the help of sklearn '''

SELECTOR = SelectKBest(mutual_info_regression, k = 4) # what are the 4 best features

SELECTOR.fit(mamoDataX,mamoDataY)

SCORES = SELECTOR.scores_
# print(SCORES)
plt.subplot(1,2,2)
plt.plot(SCORES) # how we can watch the best are [P1.Std1,P2,Std2,N2]
plt.xticks(np.arange(9),list(mamoDataX.columns))
plt.title("Score")
plt.show()

''' Now we are gonna know if those features improve the score of the model'''

Features1 = ['P1','P3','Std3','N3']
Features2 = ['P1','Std1','P2','Std2','N2']

X1_train, X1_test, Y1_train, Y1_test = X_train[Features1], X_test[Features1], Y_train, Y_test
X2_train, X2_test, Y2_train, Y2_test = X_train[Features2], X_test[Features2], Y_train, Y_test

modelTest1 = Lasso()
modelTest2 = Lasso()

modelTest1.fit(X1_train,Y1_train)
modelTest2.fit(X2_train,Y2_train)

print("originaModel: {}" .format(model.score(X_test,Y_test)))
print("modelTest1: {}" .format(modelTest1.score(X1_test,Y1_test)))
print("modelTest2: {}" .format(modelTest2.score(X2_test,Y2_test))) # the best result

newMamoData = mamoData[Features2 + ['Tissue']]
# print(newMamoData)
newMamoData.to_csv('../csv/newMamoData.csv', index = False) # we save the data more important to the model