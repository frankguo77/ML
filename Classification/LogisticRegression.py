import numpy as np
import pandas as pd
import math
import os



def read_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep = ',', header = 0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep = ',', header = 0)
    Y_train = pd.read_csv(Y_train.values)
    X_test = pd.read_csv(train_label_path, sep = ',', header = 0)
    X_test = pd.read_csv(X_test.values)
    return (X_train,Y_train,X_test)

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1*math.e^-8, 1-(1*math.e^-8))

def gradient(X,Y,Theta):
    X = np.concatenate(np.ones(1, X.shape[0]), X, axis = 1)
    vsigmoid = np.vectorize(sigmoid)
    Hypo = vsigmoid(np.dot(X, Theta))
    Error = Y - Hypo
    gradient = np.dot(X.T, Error)
    return gradient

def logistic_regression(X,Y):
    Theta = np.zeros(X.shape[1] + 1) #��0 + 0*x1 + 0*x2������ʼ�O��
    limit = 10 #����ʮ����ͣ��
    eta = 0.1 #���·���
    #costs = [] #�o�ÿ�θ��������µ�cost�Ƕ���
    for i in range(limit):
        #print("current_cost=",current_cost)
        #costs.append(current_cost)
        Theta = Theta - eta * gradient(X,Y,Theta)
        eta *= 0.95 #���·��ȣ����f�p
    #����cost��׃������������ԓҪ�ǲ����f�p �������_
    '''
    plt.plot(range(limit), costs)
    plt.show()
    '''
    return Theta

'''
define train(X, Y, save_dir):
    w = np.zeros((X_all.shape[1],))
    b = np.zeros((1,))
    l_rate = 0.1
'''

#os.path.abspath('E:/��־/ML/Classification/Data/X_train.csv')

X ,Y , Z = read_data(os.path.abspath('E:/��־/ML/Classification/Data/X_train.csv') , os.path.abspath('E:/��־/ML/Classification/Data/Y_train.csv'),os.path.abspath('E:/��־/ML/Classification/Data/X_test.csv'))
