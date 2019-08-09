# -*- coding: utf-8 -*-

# Mega Case Study - Make a Deep Learning Model

# Part 1 - Identify the Fraude with the Self-Organization Map

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv('Credit_Card_Applications.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(0, 1))
X = mms.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, 
              learning_rate=0.5, 
              neighborhood_function='gaussian')
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2, 2)], mappings[(2, 1)]), axis=0)
frauds = mms.inverse_transform(frauds)


# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = df.iloc[:, 1:-1].values

# Creating the dependent variable
is_fraud = np.zeros(len(df))
for i in range(len(df)):
    if df.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_shape=(14,)))

#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the probalilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((df.iloc[:, :1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
