# -*- coding: utf-8 -*-

# Boltzmann Machine

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', 
                     header=None, engine='python', 
                     encoding='latin-1',
                     names=['id', 'name', 'genre'])
users = pd.read_csv('ml-1m/users.dat', sep='::',
                    header=None, engine='python',
                    encoding='latin-1',
                    names=['id', 'genre', 'age', 'code', 'postal_code'])
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::',
                      header=None, engine='python',
                      encoding='latin-1', 
                      names=['id_user', 'id_movie', 'rating', 'timestamp'])

# Preparing the Training set and Test set
'''
def convert_pivot(dataframe, **kwargs):
    #opcional
    data = dataframe.pivot(index=kwargs.get('index'),
                           columns=kwargs.get('columns'),
                           values=kwargs.get('values')).fillna(0)
    return np.array(data, dtype='int')
'''
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set.iloc[:, :-1], dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set.iloc[:, :-1], dtype='int')

# Getting the number of users and movies
nb_users = len(np.unique(np.concatenate([training_set[:, 0], test_set[:, 0]])))
nb_movies = len(np.unique(np.concatenate([training_set[:, 1], test_set[:, 1]])))

# Converting the data into an array with users in lines and movies in columns
'''
def convert(data):

    matrix=np.zeros([nb_users,nb_movies],dtype="int")

    for row in data:

        matrix[row[0]-1,row[1]-1]=row[2]# 索引 size - 1

    return matrix
'''
'''
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        list = []
        for id_movie in range (1 ,nb_movies + 1):
            if id_movie in data[1]:
                list.append(data[id_movie,2])
            else:
                list.append(0)
        new_data.append(list)
    return new_data
'''
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(np.int64(ratings)))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM:
    
    def __init__(self, nv, nh):
        '''
            Informações:
                nv - Nó visível
                nh - Nó oculto
            torch.randn(tamanho do lote, Biés)
        '''
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        '''
            Informações:
                v0 -> vetor da camada de entreda
                vk -> Nó visivel após a amostragem K
                ph0 -> vetor de probabilidades que na primeira iteração os nós ocultos são igual a um
                phk -> vetor de probabilidade dos nós ocultos após a amostragem K
        '''
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


nv = len(training_set[0])
nh = 100
batch_size = 100

rbm = RBM(nv, nh)

# Training the RBM
# https://www.udemy.com/deeplearning/learn/lecture/6895698#questions/3114630
epochs = 100
for epoch in range(1, epochs + 1):
    train_loss = 0
    s = 0
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size]
        v0 = training_set[id_user:id_user + batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) ## Média das distância
        #train_loss += np.sqrt(torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])**2)) # RMSE
        s += 1
    print('epoch: {} loss: {}'.format(str(epoch), str(train_loss/s)))

# Testing the RBM
test_loss = 0
s = 0
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) # Média das distâncias
        #test_loss += np.sqrt(torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])**2)) # RMSE
        s += 1
print('test loss {}'.format(test_loss/s))


#Se você quiser verificar que 0,25 corresponde a 75% de sucesso, 
# você pode executar o seguinte teste:
u = np.random.choice([0,1], 100000)
v = np.random.choice([0,1], 100000)
u[:50000] = v[:50000]
sum(u==v)/float(len(u)) # -> you get 0.75
np.mean(np.abs(u-v)) # -> you get 0.25
#np.sqrt(np.mean(np.abs(u-v)**2)) # -> you get 0.25




