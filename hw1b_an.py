import numpy as np
import matplotlib.pyplot as plt
import sys


def lnreg_sim(m, n, verror, true_w):

    if n != len(true_w):
        raise Exception("Length of parameters vector must match the number of variables")
    
    # generating data
    X = np.random.randn(m, n)  # m independent vectors of n variables
    error = verror * np.random.randn(m)
    y = np.dot(X, true_w) + error  # response variable

    # Normal Equations
    if np.linalg.cond(X) < 1 / sys.float_info.epsilon:
        train_w_norm = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        n_error = (np.linalg.norm(true_w - train_w_norm) / np.linalg.norm(true_w)) ** 2
    else:
        return "singular matrix"

    return n_error


def main():

    true_parameters = np.asarray(range(1, 101))
    for i in true_parameters:
        true_parameters[i-1] *= (-1)**i 
    
    var_error = np.linalg.norm(true_parameters) ** 2 * 0.01

    # plotting m vs normalized error
    m = [80, 100, 120, 400]
    n_error = []
    for i in m:
        n_error.append(lnreg_sim(i, 100, var_error, true_parameters))

    print(n_error)
    plt.plot(m, n_error)
    plt.show()
 

if __name__ == "__main__":
    main()

