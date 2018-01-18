import numpy as np
def naive_pla(x,y):
    extend_x = np.concatenate((np.ones((x.shape[0], 1)), x) ,axis=1)
    w = x[0]
    
    iteration = 0
    while True:
        iteration += 1
        false_data = 0
        for i in range(x.shape[0]):
            t = np.dot(w, x[i])
            if np.sign(t) != np.sign(y[i]):
                false_data += 1
                w += y[i] * x[i]
        print('iter%d (%d / %d)' % (iteration, false_data, len(x)))
        if not false_data:
            break
    return w
            
trainData=np.array([[1, 4],  
           [2, 3],
           [-2, 3], 
           [-2, 2],
           [0, 1], 
           [1, 2]])  
label=np.array([1, 1, 1, -1, -1,  -1])  

naive_pla(trainData, label)