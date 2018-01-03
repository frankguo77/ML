import numpy as np

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
     xTrans = x.transpose()
     print(xTrans)
     #print(x)
     for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        #print(hypothesis)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
     return theta

X = np.array([[41.9,43.4,43.9,44.5,47.3,47.5,47.9,50.2,52.8,53.2,56.7,57.0,63.5,65.3,71.1,77.0,77.8], [29.1,29.3,29.5,29.7,29.9,30.3,30.5,30.7,30.8,30.9,31.5,31.7,31.9,32.0,32.1,32.5,32.9]])
#X = np.array([41.9,43.4,43.9,44.5,47.3,47.5,47.9,50.2,52.8,53.2,56.7,57.0,63.5,65.3,71.1,77.0,77.8])
y = np.array([251.3,251.3,248.3,267.5,273.0,276.5,270.3,274.9,285.0,290.0,297.0,302.5,304.5,309.3,321.7,330.7,349.0])
n = np.max(X.shape)
#print(n)
x = np.vstack([np.ones(n), X]).T
#print(x)
m, n = np.shape(x)
print(m,n)
numIterations= 100000
alpha = 0.000005
theta = np.ones(n)
print(theta)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)