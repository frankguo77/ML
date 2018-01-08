import numpy as np
import csv
import math

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, numIterations):
    print('gradientDescent:')
    xTrans = x.transpose()
    #xTrans = x.T
    # print(xTrans)
    print(x.shape)
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        #print(hypothesis.shape)
        loss = hypothesis - y
       #print(loss.shape)
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2)/len(x)
        # print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss)
        #print(gradient.shape)
        # update
        theta = theta - alpha * gradient
        print('iteration: %d | Cost: %f  ' % (i, cost))
    return theta

def adagraDescent(x, y, w, alpha, numIterations):
    print('AdagradientDescent:')
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))

    for i in range(numIterations):
        hypo = np.dot(x, w)
        loss = hypo - y
        cost = np.sum(loss ** 2) / len(x)
        cost_a = math.sqrt(cost)
        gra = np.dot(x_t, loss)
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra / ada
        print('iteration: %d | Cost: %f  ' % (i, cost_a))
    return w

def readTainningData():
    data = []
    # 每一個維度儲存一種污染物的資訊
    for i in range(18):
        data.append([])

    n_row = 0
    text = open('Data/train.csv', 'r', encoding='big5')
    row = csv.reader(text, delimiter=",")
    for r in row:
        # 第0列沒有資訊
        if n_row != 0:
            # 每一列只有第3-27格有值(1天內24小時的數值)
            for i in range(3, 27):
                if r[i] != "NR":
                    data[(n_row - 1) % 18].append(float(r[i]))
                else:
                    data[(n_row - 1) % 18].append(float(0))
        n_row = n_row + 1
    text.close()
    return data

def parseData(data):
    x = []
    y = []
    # 每 12 個月
    for i in range(12):
        # 一個月取連續10小時的data可以有471筆
        for j in range(471):
            x.append([])
            # 18種污染物
            for t in range(18):
                # 連續9小時
                for s in range(9):
                    x[471 * i + j].append(data[t][480 * i + j + s])
            y.append(data[9][480 * i + j + 9])
    x = np.array(x)
    y = np.array(y)

    # add square term
    # x = np.concatenate((x,x**2), axis=1)

    # add bias
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    print(x.shape)

    return x,y

def readTestData():
    test_x = []
    n_row = 0
    text = open('data/test.csv', "r")
    row = csv.reader(text, delimiter=",")

    for r in row:
        if n_row % 18 == 0:
            test_x.append([])
            for i in range(2, 11):
                test_x[n_row // 18].append(float(r[i]))
        else:
            for i in range(2, 11):
                if r[i] != "NR":
                    test_x[n_row // 18].append(float(r[i]))
                else:
                    test_x[n_row // 18].append(0)
        n_row = n_row + 1
    text.close()
    test_x = np.array(test_x)

    # add square term
    # test_x = np.concatenate((test_x,test_x**2), axis=1)

    # add bias
    test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)
    return test_x

def saveResult(resulstfile,test_x,theta):
    print(theta)
    ans = []
    for i in range(len(test_x)):
        ans.append(["id_" + str(i)])
        a = np.dot(theta, test_x[i])
        ans[i].append(a)

    filename = "result/" + resulstfile
    text = open(filename, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "value"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

rawdata = readTainningData()
#print(rawdata)

x, y = parseData(rawdata)

theta = np.zeros(len(x[0]))
#print(theta.shape)
l_rate = 0.00000001
repeat = 10000

#theta = gradientDescent(x, y, theta, l_rate,repeat)
test_x = readTestData()
print(test_x)
print(test_x.dtype)

#saveResult('normalResult.csv',test_x)


theta = np.zeros(len(x[0]))
#print(theta.shape)
theta = adagraDescent(x, y, theta, l_rate,repeat)

saveResult('adaresult.csv',test_x,theta)

