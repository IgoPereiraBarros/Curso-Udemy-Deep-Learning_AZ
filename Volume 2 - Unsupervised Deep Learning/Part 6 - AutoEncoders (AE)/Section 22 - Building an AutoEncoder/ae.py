# -*- coding: utf-8 -*-

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
                    names=['id', 'genre', 'age', 'codes', 'postal_code'])
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

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    
    def __init__(self, nb_movies):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE(nb_movies)
criterion = nn.MSELoss()
optimazer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE
epochs = 200
for epoch in range(1, epochs + 1):
    train_loss = 0
    s = 0
    for id_user in range(nb_users):
        _input = Variable(training_set[id_user]).unsqueeze(0)
        target = _input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(_input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1
            optimazer.step()
    print('epoch: {} Training loss: {}'.format(str(epoch), str(train_loss/s)))


# Testing the SAE
test_loss = 0
s = 0
for id_user in range(nb_users):
    _input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(_input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1
print('Test loss: {}'.format(str(test_loss/s)))




# Complemento
'''
#save the training model
torch.save(sae, 'AutoEncoder.pkl')#you can use one of the function to save the training model
#torch.save(sae.state_dict(), 'AutoEncoder.pkl')
 
#next time you can using the following code to load the model you've trained, just replace 'sae = SAE()' with following code, and delete every lines of code behind (loss computation)
#sae = torch.load('AutoEncoder.pkl')
 
from operator import itemgetter#用来使用sorted() function
#prediction
#定义prediction function
def Prediction(user_id, nb_recommend):
    user_input = Variable(test_set[user_id - 1]).unsqueeze(0)
    predict_output = sae.predict(user_input)
    predict_output = predict_output.data.numpy()
    predicted_result = np.vstack([user_input, predict_output])
 
    trian_movie_id = np.array([i for i in range(1, nb_movies+1)])#create a temporary index for movies since we are going to delete some movies that the user had seen, 创建一个类似id的index，排序用
    recommend = np.array(predicted_result)
    recommend = np.row_stack((recommend, trian_movie_id))#insert that index into the result array, 把index插入结果
    recommend = recommend.T#transpose row and col 数组的行列倒置
    recommend = recommend.tolist()#tansfer into list for further process转化为list以便处理
 
    movie_not_seen = []#delete the rows comtaining the movies that the user had seen 删除users看过的电影
    for i in range(len(recommend)):
        if recommend[i][0] == 0.0:
            movie_not_seen.append(recommend[i])
 
    movie_not_seen = sorted(movie_not_seen, key=itemgetter(1), reverse=True)#sort the movies by mark 按照预测的分数降序排序
 
    recommend_movie = []#create list for recommended movies with the index we created 推荐的top20
    for i in range(0, nb_recommend):
        recommend_movie.append(movie_not_seen[i][2])
 
    recommend_index = []#get the real index in the original file of 'movies.dat' by using the temporary index这20部电影在原movies文件里面真正的index
    for i in range(len(recommend_movie)):
        recommend_index.append(movies[(movies.iloc[:,0]==recommend_movie[i])].index.tolist())
 
    recommend_movie_name = []#get a list of movie names using the real index将对应的index输入并导出movie names
    for i in range(len(recommend_index)):
        np_movie = movies.iloc[recommend_index[i],1].values#transefer to np.array
        list_movie = np_movie.tolist()#transfer to list
        recommend_movie_name.append(list_movie)
 
    print('Highly Recommended Moives for You:\n')
    for i in range(len(recommend_movie_name)):
        print(str(recommend_movie_name[i]))
    
    return recommend_movie_name
 
#recommendation for target user's id
user_id = 367
#the number of movies recommended for the user
nb_recommend = 20
movie_for_you = Prediction(user_id = user_id, nb_recommend = nb_recommend)
'''


