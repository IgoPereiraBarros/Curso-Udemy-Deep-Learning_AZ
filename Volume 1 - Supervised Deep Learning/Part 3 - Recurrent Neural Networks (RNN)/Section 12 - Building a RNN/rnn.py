# -*- coding: utf-8 -*-

# Part 1 - Data Preprocessing

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling (Normalização, Padronização)
# OBS: Irei testar padronização nos dados, usando o StandardScaler
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = mms.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i])
    y_train.append(training_set_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(.2))

# Adding a secund LTSM layer and some Dropout regularization
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(.2))

# Adding a third LTSM layer and some Dropout regularization
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(.2))

# Adding a third LTSM layer and some Dropout regularization
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(.2))

# Adding a third LTSM layer and some Dropout regularization
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(.2))

# Adding a fourth LTSM layer and some Dropout regularization
regressor.add(LSTM(units=100))
regressor.add(Dropout(.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# Part 3 - Making the predictions and Visualizing the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train.Open, dataset_test.Open), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = mms.fit_transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = mms.inverse_transform(predicted_stock_price)

# Visualising the result
import seaborn as sns
sns.set_style('darkgrid')

plt.plot(real_stock_price, color='r', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='b', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

# Improving the RNN

'''
Aqui estão diferentes maneiras de melhorar o modelo RNN:

1 - Obtendo mais dados de treinamento: treinamos nosso modelo nos últimos 5 anos do Google Stock Price, 
    mas seria ainda melhor treiná-lo nos últimos 10 anos.

2 - Aumentando o número de prazos: o modelo lembrou os preços das ações dos 60 dias financeiros anteriores para prever o preço das ações do dia seguinte. 
    Isso porque escolhemos um número de 60 timesteps (3 meses).
    Você poderia tentar aumentar o número de timesteps, 
    escolhendo por exemplo 120 timesteps (6 meses). (TESTADO - MODELO PIOROU)

3 - Adicionando alguns outros indicadores: se você tiver o instinto financeiro de que o preço das ações de algumas outras empresas possa estar correlacionado ao do Google, 
    você pode adicionar esse outro preço de ação como um novo indicador nos dados de treinamento.

4 - Adicionando mais camadas LSTM: construímos um RNN com quatro camadas LSTM, 
    mas você pode tentar ainda mais.(TESTADO - NÃO HOUVE NENHUMA MELHORIA)

5 - Adicionando mais neurônios nas camadas LSTM: destacamos o fato de que precisávamos de um número elevado de neurônios nas camadas LSTM para responder melhor à complexidade do problema e optamos por incluir 50 neurônios em cada uma das nossas 4 camadas LSTM. 
    Você poderia tentar uma arquitetura com ainda mais neurônios em cada uma das 4 (ou mais) camadas LSTM. (TESTADO - O MODELO MELHOROU)
'''


# Turning the RNN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_classifier(optimizer):
    regressor = Sequential()
    regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(.2))
    
    regressor.add(LSTM(units=100, return_sequences=True))
    regressor.add(Dropout(.2))
    
    regressor.add(LSTM(units=100, return_sequences=True))
    regressor.add(Dropout(.2))
    
    regressor.add(LSTM(units=100, return_sequences=True))
    regressor.add(Dropout(.2))
    
    regressor.add(LSTM(units=100, return_sequences=True))
    regressor.add(Dropout(.2))
    
    regressor.add(LSTM(units=100))
    regressor.add(Dense(units=1))
    
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return regressor

regressor = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']
              }
grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=10)
grid_search = grid_search.fit(X_train, y_train) 

best_parameters = grid_search.best_estimator_
best_accuracy = grid_search.best_score_











