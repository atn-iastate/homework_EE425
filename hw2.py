import scipy.stats as st
import numpy as np


class GDAModel:

    def __init__(self, mu0, mu1, cov, phi):
        self.mu0 = mu0
        self.mu1 = mu1
        self.cov = cov
        self.phi = phi

# def logistic_regression(x, y):


def compute_variance(x, y, mu0, mu1):

    if y.shape[0] > 0:
        var = (np.sum((x[np.where(y == 0)] - mu0)**2) + np.sum((x[np.where(y == 1)] - mu1)**2)) / y.shape[0]
        return var
    else:
        return None


def gda_estimate(x, y):

    if y.shape[0] == 0:
        return GDAModel(None, None, None, None)
    else:
        estimate_phi = np.sum(y) / y.shape[0]

    if np.sum(y) < y.shape[0]:
        estimate_mu0 = np.sum(x[np.where(y == 0), :], 1) / np.where(y == 0)[0].shape[0]
    else:
        estimate_mu0 = None

    if np.sum(y) > 0:
        estimate_mu1 = np.sum(x[np.where(y == 1), :], 1) / np.where(y == 0)[0].shape[0]
    else:
        estimate_mu1 = None

    # initialize covariance matrix with zeroes entries of dimension n x n
    estimate_cov = np.zeros(x.shape[1] ** 2).reshape(x.shape[1], x.shape[1])
    for i in range(x.shape[1]):
        estimate_cov[i, i] = compute_variance(x[:, i], y, estimate_mu0[i], estimate_mu1[i])

    model = GDAModel(estimate_mu0, estimate_mu1, estimate_cov, estimate_phi)

    return model


# generate training data
phi = 0.4
m = 100
m_test = 25
n = 10
sigma = np.random.uniform(5, 15, n)
mu0 = np.random.uniform(0, 10, n)
mu1 = np.random.uniform(0, 10, n)
y = st.bernoulli.rvs(phi, size=m)
cov = np.diag(sigma)
x = np.zeros(m*n).reshape(m, n)
for i in range(m):
    mu = mu0 * (1 - y[i]).item() + mu1 * y[i].item()
    x[i, :] = np.random.multivariate_normal(mu, cov)

y_test = y = st.bernoulli.rvs(phi, size=m_test)

x_test = np.random.randn(m_test, n).reshape(m_test, n)
for i in range(m_test):
    mu = mu0 * (1 - y_test[i]).item() + mu1 * y_test[i].item()
    x_test[i, :] = np.random.multivariate_normal(mu, cov)

