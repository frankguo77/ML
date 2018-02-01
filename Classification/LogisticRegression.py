import numpy as np
import pandas as pd
import math
import os


def read_data(train_data_path, train_label_path, test_data_path):
    f = open(train_data_path)
    X_train = pd.read_csv(f, sep=',', header=0)
    f.close()
    X_train = np.array(X_train.values)
    f = open(train_label_path)
    Y_train = pd.read_csv(f, sep=',', header=0)
    f.close()
    Y_train = np.array(Y_train.values)
    f = open(test_data_path)
    X_test = pd.read_csv(f, sep=',', header=0)
    f.close()
    X_test = np.array(X_test.values)
    return (X_train, Y_train, X_test)


def sigmoid(z):
    print('z=', z)
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1 * math.e ** -8, 1 - (1 * math.e ** -8))


def normalize(X):
    mean = sum(X) / X.shape[0]
    sigma = np.std(X, axis=0)
    mean = np.tile(mean, (X.shape[0], 1))
    sigma = np.tile(sigma, (X.shape[0], 1))
    X = (X - mean) / sigma
    return X


def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def gradient(X, Y, Theta):
    vsigmoid = np.vectorize(sigmoid)
    Hypo = vsigmoid(np.dot(X, Theta))
    # print("Hypo.shape = ", Hypo.shape)
    # Hypo.reshape((len(Hypo), 1))
    Error = Y - Hypo
    gra = np.dot(X.T, Error)
    return gra


def logistic_regression(X, Y, Theta):
    eta = 0.1  # 更新幅度
    Theta = Theta - eta * gradient(X, Y, Theta)
    return Theta


def valid(Theta, X_valid, Y_valid):
    np.concatenate((np.ones((X_valid.shape[0], 1)), X_valid), axis=1)
    valid_data_size = len(X_valid)
    Hypo = vsigmoid(np.dot(X_valid, Theta))
    # print("Hypo.shape = ", Hypo.shape)
    # Hypo.reshape((len(Hypo), 1))
    Error = Y - Hypo
    Error_ = np.around(Error)
    result = (np.squeeze(Y_valid) == Error_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return


def train():
    X, Y, Z = read_data(r'E:/郭志/ML/Classification/Data/X_train.csv', r'E:/郭志/ML/Classification/Data/Y_train.csv',
                        r'E:/郭志/ML/Classification/Data/X_test.csv')
    X = normalize(X)
    Y = Y.flatten()
    batch_size = 32
    train_data_size = len(X)
    step_num = int(math.floor(train_data_size / batch_size))
    epoch_num = 1
    save_param_iter = 50

    Theta = np.zeros(X.shape[1] + 1)  # 用0 + 0*x1 + 0*x2當作初始設定
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            '''
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            '''
            valid(Theta, X_valid, Y_valid)
        # Random shuffle

    X, Y = shuffle(X, Y)

    # Train with batch
    for idx in range(step_num):
        X_batch = X[idx * batch_size:(idx + 1) * batch_size]
        Y_batch = Y[idx * batch_size:(idx + 1) * batch_size]
        X_batch = np.concatenate((np.ones((X_batch.shape[0], 1)), X_batch), axis=1)
        Theta = logistic_regression(X_batch, Y_batch, Theta)


    return

train()
