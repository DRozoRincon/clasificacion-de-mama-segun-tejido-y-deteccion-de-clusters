import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso

mamoDataX = pd.read_csv('../csv/prueba.csv')
mamoDatay = mamoDataX['Tissue']
mamoDataX = mamoDataX.drop('Tissue',axis=1)

'''
    El metodo train_test_split es un metodo utilizado para entrenar nestros datos sin embargo es un metodo
    para hacer mas que nada prototipos debido a que el score que se puede obtener de este no es muy confiable 
    por eso solo se uutiliza para el estudio de el modelo sin embargo cuando queremos entrenar los datos para
    modelo fijo y final utilizaremos cross validate debido a que entrena los datos de una manera mas robusta
    aunque el score pueda ser mas bajo que con train_test_split, pero es mas confiable
'''

X_train, X_test, Y_train, Y_test = train_test_split(mamoDataX,mamoDatay, random_state = 1) # los datos de testeo y train siempre cambia cada vez que se corre el progrma por eso usamos random_state = # para que no suceda esto

scores = cross_val_score(Lasso(),mamoDataX,mamoDatay,cv=5,scoring='r2') # cv (cross values, 5 por que son el numero de folds comparacion ... (test,train))

print("score got with croo_val: {}" .format(scores))
print("mean score: {}" .format(scores.mean())) # para saber el score promedio  