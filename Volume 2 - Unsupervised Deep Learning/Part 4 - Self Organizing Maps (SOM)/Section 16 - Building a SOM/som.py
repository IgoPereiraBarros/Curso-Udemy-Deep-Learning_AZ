# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv('Credit_Card_Applications.csv')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(0, 1))
X = mms.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=14, 
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

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2, 2)], mappings[(8, 5)]), axis=0)
frauds = mms.inverse_transform(frauds)

# Prediting a unique client
pred = mms.inverse_transform(X)
max_dist = 0.9
w = som.winner(pred[90])
if som.distance_map()[w[0], w[1]] > max_dist:
    print(pred[90], ' é uma fraude')
else:
    print('não é fraude')

# Ids e coordenadas das clientes fraudufela da puta
ids = df.iloc[:, 0].values
max_dist = 0.5
fraud_IDs = []
fraud_locations = []
for i, x in enumerate(X):
    w = som.winner(x)
    if som.distance_map()[w[0], w[1]]>max_dist:
        fraud_locations.append(w)
        fraud_IDs.append(ids[i])
