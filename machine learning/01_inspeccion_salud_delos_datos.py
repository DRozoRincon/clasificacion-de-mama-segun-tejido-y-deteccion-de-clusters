import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.impute import SimpleImputer

mamoData = pd.read_csv("../csv/mamoDataX1.csv") 
normalData = mamoData['Tissue']

# Data inspection #

print(mamoData.info()) # to knoe general info
# mamoData = mamoData.drop('Unnamed: 0',axis=1)
print(mamoData.describe()) # to know statistics info
mamoData[['Tissue']].hist() # hist info to each one
print(mamoData)
mamoData = mamoData.drop('Tissue',axis = 1)

# Zero data inspection #

print((mamoData != 0).apply(pd.Series.value_counts)) # to know how many data in each feature comply condition (there are a lot of zeros in three last features)

dataAvailable = (mamoData != 0) 
print(dataAvailable.all(axis=1).value_counts) # to know what rows comply the condition

# We are gonna put the mean of the each feature in each value 0 of that feature (if maybe betterment the model score) #

imputer = SimpleImputer(missing_values = 0,strategy = 'mean')
noneZeros = imputer.fit_transform(mamoData) # assignating the strategy in each feature

# Now the data is a array we are gonna convet it in DataFrame

noneZeros = pd.DataFrame(noneZeros)
noneZeros.columns = mamoData.columns # assignating columns
noneZeros.index = mamoData.index # assignating index

noneZeros = pd.concat([noneZeros,normalData], axis = 1) # concating the Tissue feature

print((noneZeros != 0).apply(pd.Series.value_counts)) # verifying the results (all correct)

# noneZeros.to_csv('../csv/mamoDataX1.csv', index = False) # creating a new csv file