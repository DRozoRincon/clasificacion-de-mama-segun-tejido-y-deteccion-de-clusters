import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate,validation_curve
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

mamoDataX = pd.read_csv('../csv/prueba.csv')
mamoDataY = mamoDataX['Tissue']
mamoDataX = mamoDataX.drop('Tissue',axis=1)


''' the idea always is chose a robust model neither so simple neither so complex '''

result_lasso = cross_validate(Lasso(), mamoDataX, mamoDataY, return_train_score = True, cv=5) # con este metodo obtenemos los datos print(Result)
# print(result_lasso)

test_score = result_lasso['test_score']
train_score = result_lasso['train_score']
print("Test score with Lasso model: {} " .format(np.mean(test_score)))
print("Train score with Lasso model: {}" .format(np.mean(train_score)))

''' 
    el modelos KNeighborsRegressor lo que hace es buscar el vecino los vecinos mas cercanos 
    y hacer un promedio de estos
'''

''' the KneighborsRegressor what it do is search (N) neighbourshodd closer and do a mean of them '''



'''Now we are gonna watch with how many Kneighbours the model predict better '''

Nvecinos = np.arange(2,50,2) # 24 datos totales
train_scores, test_scores =validation_curve(KNeighborsRegressor(),mamoDataX,mamoDataY,param_name='n_neighbors',param_range=Nvecinos,cv=5) # obtenemos los datos de score (train,test) para los distintos cantidad de vecino (2-50)

#graficamos los resultados score vs vecinos
plt.plot(np.mean(train_scores,axis=1)) 
plt.plot(np.mean(test_scores,axis=1))
plt.xticks(np.arange(24),Nvecinos)

''' How we can watch thebest (n) is 46 '''

result_neighbour = cross_validate(KNeighborsRegressor(n_neighbors=46), mamoDataX, mamoDataY, return_train_score = True, cv=5) 

test_score2 = result_neighbour['test_score']
train_score2 = result_neighbour['train_score']
print(np.mean(test_score2))
print(np.mean(train_score2))
