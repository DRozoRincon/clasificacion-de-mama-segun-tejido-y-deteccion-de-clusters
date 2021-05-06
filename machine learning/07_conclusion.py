import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

mamoDataX = pd.read_csv('../csv/prueba.csv')
mamoDataY = mamoDataX['Tissue']
mamoDataX = mamoDataX.drop(['Tissue'], axis = 1)


''' 
    the conclusion of all this is that with little data is better use linear model
    but when we have a lot of data is better use model like forest and gradient tree decision-based 
'''
    

X_train, X_test, Y_train, Y_test = train_test_split(mamoDataX,mamoDataY, random_state = 2)

model = KNeighborsRegressor(18)
model2 = Lasso()


model.fit(X_train,Y_train)
model2.fit(X_train,Y_train)

print(model.score(X_test,Y_test))
print(model2.score(X_test,Y_test))


PREDICTED = model.predict(X_test)
PREDICTED2 = model2.predict(X_test)

plt.subplot(2,2,1)
plt.hist([PREDICTED,Y_test])

plt.subplot(2,2,2)
plt.hist([PREDICTED2,Y_test])

plt.show