from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

def LoadData():
    concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
    concrete_data.head()
    return concrete_data


def regression_model(n_cols):
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def RegressionModel():
    concrete_data = LoadData()
    concrete_data_columns = concrete_data.columns

    predictors = concrete_data[
        concrete_data_columns[concrete_data_columns != 'Strength']]
    target = concrete_data['Strength']

    predictors_norm = (predictors - predictors.mean()) / predictors.std()
    predictors_norm.head()
    n_cols = predictors_norm.shape[1]

    model = regression_model(n_cols)
    model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)