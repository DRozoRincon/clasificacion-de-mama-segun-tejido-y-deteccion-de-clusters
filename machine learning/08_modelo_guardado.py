import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.externals import joblib # to save our model
from sklearn.pipeline import make_pipeline # to do various thing in a model
from sklearn.preprocessing import PolynomialFeatures # to create the features
from collections import OrderedDict


mamoDataX = pd.read_csv('../csv/newMamoData.csv')
mamoDataY = mamoDataX['Tissue']
mamoDataX = mamoDataX.drop(['Tissue'], axis = 1)

''' 
    How we experimented the best election is a linear model and creating features for that we are gonna save that
    model to use it in our project 
'''

X_train, X_test, Y_train, Y_test = train_test_split(mamoDataX, mamoDataY, random_state = 2)

mamoModel = make_pipeline(PolynomialFeatures(2), Lasso())
mamoModel.fit(X_train,Y_train)

print(mamoModel.score(X_test,Y_test))  # predict percent

# mamoFeatures = OrderedDict([
#         ('P1', 500),
#         ('Std1', 200),
#         ('P2',150),
#         ('Std2', 100),
#         ('N2', 0.4)
#     ])
    
# mamoFeatures = pd.Series(mamoFeatures).values.reshape(1,-1)
    
# predict = mamoModel.predict(mamoFeatures)

# joblib.dump(mamoModel, 'mammography_model.pkl')