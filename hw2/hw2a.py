import scipy.stats as st
import numpy as np
import math


class GDAModel:

    def __init__(self, mu0, mu1, phi):
        self.mu0 = mu0
        self.mu1 = mu1
        self.phi = phi
        self.no_of_feature = mu0.shape[0]

    def predict(self, x):

        y_predict = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if np.linalg.norm(x[i,:] - self.mu0) > np.linalg.norm(x[i,:] - self.mu1):
                y_predict[i] = 1

        return y_predict

    @staticmethod
    def gda_estimate(x, y):

        if y.shape[0] == 0:
            return GDAModel(None, None, None)
        else:
            estimate_phi = np.sum(y) / y.shape[0]

        if np.sum(y) < y.shape[0]:  # will only work if y is Bernoulli distributed
            estimate_mu0 = np.sum(x[np.where(y == 0)[0], :], axis=0) / np.where(y == 0)[0].shape[0]
        else:
            estimate_mu0 = None

        if np.sum(y) > 0:
            estimate_mu1 = np.sum(x[np.where(y == 1)[0], :], axis=0) / np.where(y == 0)[0].shape[0]
        else:
            estimate_mu1 = None

        # initialize covariance matrix with zeroes entries of dimension n x n
        estimate_cov = np.zeros(x.shape[1] ** 2).reshape(x.shape[1], x.shape[1])

        model = GDAModel(estimate_mu0, estimate_mu1, estimate_phi)

        return model


class LogisticModel:

    def __init__(self, parameters):
        self.parameters = parameters

    @staticmethod
    def logistic_estimate(x, y, max_iter):
        learn_rate = (1e-2) / y.shape[0]
        theta_hat = np.ones(x.shape[1])

        for t in range(max_iter):
            hx = (1 / (1 + np.exp(-np.dot(x, theta_hat))))
            theta_hat = theta_hat - learn_rate * (np.dot(np.transpose(x), (hx - y)))

        model = LogisticModel(theta_hat)
        return model

    def predict(self, x):

        hx = (1 / (1 + np.exp(-np.dot(x, self.parameters))))
        y_hat = np.zeros(hx.shape[0])

        for i in range(0, len(hx)):
            if hx[i] >= 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat


def compute_accuracy(y_test, y_hat):
    accuracy = 1 - ((1 / y_test.shape[0]) * np.sum(np.abs(y_test - y_hat)))
    return accuracy


def main():
    # parameters for the model
    phi = 0.4
    m = 20
    m_test = 100
    n = 100
    sigma = np.random.uniform(5, 15, n)
    mu0 = np.random.uniform(0, 10, n)
    mu1 = np.random.uniform(0, 10, n)

    # generate training data
    y = st.bernoulli.rvs(phi, size=m)
    cov = np.diag(sigma)
    x = np.zeros(m * n).reshape(m, n)
    for i in range(m):
        mu = mu0 * (1 - y[i]).item() + mu1 * y[i].item()
        x[i, :] = np.random.multivariate_normal(mu, cov)

    # generate test data
    y_test = st.bernoulli.rvs(phi, size=m_test)

    x_test = np.random.randn(m_test, n).reshape(m_test, n)
    for i in range(m_test):
        mu = mu0 * (1 - y_test[i]).item() + mu1 * y_test[i].item()
        x_test[i, :] = np.random.multivariate_normal(mu, cov)

    # Estimate using GDA and compute accuracy
    gda_model = GDAModel.gda_estimate(x, y)

    y_hat = gda_model.predict(x_test)
    gda_accuracy = compute_accuracy(y_test, y_hat)

    print("GDA Model accuracy is: %r" % gda_accuracy)

    # Estimate using Logistic regression and compute accuracy
    logistic_model = LogisticModel.logistic_estimate(x, y, max_iter=1000)
    y_hat = logistic_model.predict(x_test)

    logistic_accuracy = compute_accuracy(y_test, y_hat)
    print("Logistic Regression Model is: %r" % logistic_accuracy)


if __name__ == "__main__":
    main()
