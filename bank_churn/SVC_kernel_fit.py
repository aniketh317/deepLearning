from multiprocessing import Manager, Process
import numpy as np
from sklearn.svm import SVC
import pandas as pd

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
multi_categorical_features = ['Geography']
binary_categorical_features = ['Gender']
pass_ordinal_features = ['NumOfProducts']
pass_cardinal_features  = ['HasCrCard', 'IsActiveMember']

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

numerical_transformer = Pipeline(steps=[
    ('standardScale', StandardScaler()) # Using standard scaling. Shift to zero mean and scale to unit variance
])

binary_categorical_transformer = Pipeline(steps=[
    ('ordinalEncode', OrdinalEncoder()) # Using ordinal encoder even for categorical here since only 2 classes(binary)
])

multi_categorical_transformer = Pipeline(steps=[
    ('onehotEncode', OneHotEncoder(sparse_output=False)) # Allowing dense representation(i.e. zeros are present too)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('bincat', binary_categorical_transformer, binary_categorical_features),
        ('multicat', multi_categorical_transformer, multi_categorical_features),
        ('pass', 'passthrough', pass_ordinal_features+pass_cardinal_features),
    ]
)

X_train = preprocessor.fit_transform(train) #output of the transformation is numpy
X_test = preprocessor.transform(test) #output of the transformation is numpy



def SVC_fit_and_prob_predict(Map, kernel='rbf'):
    model = None
    if kernel=='rbf':
        model = SVC(kernel='rbf', gamma='scale', probability=True) # Gaussian Kernel
    elif kernel=='poly':
        model = SVC(kernel='poly', degree=3, gamma='scale', coef0=1, probability=True) # Polynomial Kernel
    elif kernel == 'sigmoid':
        model = SVC(kernel='sigmoid', gamma='scale', coef0=0, probability=True) # Sigmoid Kernel
    else:
        model = SVC(kernel='linear', probability=True) # Linear Kernel
    
    print(kernel)
    model.fit(X_train, train['Exited'])
    y_predict_test = model.predict_proba(X_test)
    Map[kernel] = y_predict_test

if __name__ == '__main__': 
    # Define the shared queue, which the process will share with its children that it intends to share with
    Map = Manager().dict()

    # Define the processes
    rbf = Process(target=SVC_fit_and_prob_predict, args=(Map, 'rbf',))
    poly = Process(target=SVC_fit_and_prob_predict, args=(Map, 'poly',))

    # Start the processes with the respective start()
    rbf.start()
    poly.start()

    #Wait for the proecsses to finish at the respective join()
    rbf.join()
    poly.join()

    dframe = pd.DataFrame({'id':test['id'], 'Exited': Map['rbf'][:,1]})
    dframe.set_index('id', inplace=True)
    dframe.to_csv('./output/rbfSVC.csv')

    dframe = pd.DataFrame({'id':test['id'], 'Exited': Map['poly'][:,1]})
    dframe.set_index('id', inplace=True)
    dframe.to_csv('./output/polySVC.csv')     