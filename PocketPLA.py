import numpy as np
def pocket_pla(x,y,limit):
    ###############
    def _calc_false(w):
        res = 0
        for i in range(len(x)):
            t = np.dot(w, x[i])
            if np.sign(y[i]) != np.sign(t):
                res += 1
        return res
    ###############

    extend_x = np.concatenate((np.ones((x.shape[0], 1)), x) ,axis=1)
    w = x[0]
    least_false = _calc_false(w)
    res = w
    
    iteration = 0
    for i in range(limit):
        iteration += 1
        false_data = 0
        for i in range(x.shape[0]):
            t = np.dot(w, x[i])
            if np.sign(t) != np.sign(y[i]):
                false_data += 1
                w += y[i] * x[i]
                t_false = _calc_false(w)
                if t_false <= least_false:
                    least_false = t_false
                    res = w
    return w

trainData=np.array([[1, 4],  
           [2, 3],
           [-2, 3], 
           [-2, 2],
           [0, 1], 
           [1, 2]])  
label=np.array([1, 1, 1, -1, -1,  -1])  

pocket_pla(trainData, label,1000)
  