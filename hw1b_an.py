import numpy as np
import matplotlib.pyplot as plt
import sys


def lnreg_sim(m, n, var_error, true_w):

    if n != len(true_w):
        raise Exception("Length of parameters vector must match the number of variables")
    
    # generating data
    X = np.random.randn(m, n)  # m independent vectors of n variables
    error = var_error * np.random.randn(m)
    y = np.dot(X, true_w) + error  # response variable

    # Normal Equations
    if np.linalg.cond(X) < 1 / sys.float_info.epsilon:  # checking if the matrix is invertible
        train_w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        n_error = (np.linalg.norm(true_w - train_w) / np.linalg.norm(true_w)) ** 2
        return n_error, train_w
    else:
        return 100000, None


def main():

    # setting up parameters for the simulation
    true_parameters = np.asarray(range(1, 101))
    for i in true_parameters:
        true_parameters[i-1] *= (-1)**i 

    n = 100
    m_test = 10000
    m = [80, 100, 120, 400]
    var_error_1 = np.linalg.norm(true_parameters) ** 2 * 0.01
    var_error_2 = np.linalg.norm(true_parameters) ** 2 * 0.1
    test_x = np.random.randn(m_test, n)
    test_y_1 = np.dot(test_x, true_parameters) + np.random.randn(m_test)*var_error_1
    test_y_2 = np.dot(test_x, true_parameters) + np.random.randn(m_test)*var_error_2

    # for each variance value, compute normalized error (n_error) and expected MSE  (mse)
    n_error_1 = []
    mse_1 = []
    for i in m:
        sim_var_1 = lnreg_sim(i, n, var_error_1, true_parameters)
        n_error_1.append(sim_var_1[0])

        estimated_w = sim_var_1[1]
        n_test_error_vector = [i ** 2 / j ** 2 for i, j in zip(np.dot(test_x, estimated_w) - test_y_1, test_y_1)]
        mse_1.append(sum(n_test_error_vector) / len(n_test_error_vector))

    n_error_2 = []
    mse_2 = []
    for i in m:
        sim_var_2 = lnreg_sim(i, n, var_error_2, true_parameters)
        n_error_2.append(sim_var_2[0])

        estimated_w = sim_var_2[1]
        n_test_error = [i ** 2 / j ** 2 for i, j in zip(np.dot(test_x, estimated_w) - test_y_2, test_y_2)]
        mse_2.append(sum(n_test_error) / len(n_test_error))

    plot, axs = plt.subplots(2, 2)

    axs[0, 0].plot(m, n_error_1)
    axs[0, 0].set_title("Normalized error against observations (v = 0.01)")
    axs[0, 1].plot(m, n_error_2)
    axs[0, 1].set_title("Normalized error against observations (v = 0.1)")
    axs[1, 0].plot(m, mse_1)
    axs[1, 0].set_title("mean squared error against observations (v = 0.01)")
    axs[1, 1].plot(m, mse_2)
    axs[1, 1].set_title("Normalized error against observations (v = 0.1)")

    plt.show()
 

if __name__ == "__main__":
    main()

