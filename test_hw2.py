import numpy as np
from hw2 import gda_estimate
from hw2 import compute_variance


def test_compute_variance():

    x = np.arange(0, 5, 1)
    y = np.asarray([0, 1, 0, 1, 0])
    mu0 = 1
    mu1 = 2
    correct_value = 13/5

    return compute_variance(x, y, mu0, mu1) == correct_value


def test_gda_estimate():
    x = np.arange(0, 10, 1).reshape(5, 2)
    y = np.asarray([0, 1, 0, 1, 0])
    mu0 = np.asarray([1, 2])
    mu1 = np.asarray([3, 4])

def main():
    # x = np.arange(0,5,1)
    # print(np.where(x % 2 == 0)[0].shape)
    y = np.asarray([0, 1, 0, 1, 0])
    print("test_compute_variance: %r" % test_compute_variance())
    print(y.shape[0])


if __name__ == '__main__':
    main()