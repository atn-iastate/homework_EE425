import numpy as np
import matplotlib.pyplot as plt


def gradient_descent_estimate(x, y, initial_w, learning_rate, max_iter):

    train_w = initial_w

    for i in range(max_iter):
        gradient = np.dot(np.dot(np.transpose(x), x), train_w) - np.dot(np.transpose(x), y)
        train_w = train_w - learning_rate * gradient

    return train_w


x = np.random.randn(100, 100)

true_parameters = np.asarray(range(1, 101))
for i in true_parameters:
    true_parameters[i - 1] *= (-1) ** i

y = np.dot(x, true_parameters) + np.random.randn(100) * 0.1

test_x = np.random.randn(40, 100)
test_y_1 = np.dot(test_x, true_parameters) + np.random.randn(40) * 0.1

mse = []

for i in range(100):
    train_w = gradient_descent_estimate(x[:, 0:i+1], y, initial_w=np.zeros(i+1), learning_rate=0.001, max_iter=1000)

    # error_vector = np.dot(x[:, 0:i+1], train_w) - y
    # test_mse = np.dot(np.transpose(error_vector), error_vector)
    mse.append((np.linalg.norm(np.dot(test_x[:, 0:i+1], train_w) - test_y_1) / np.linalg.norm(test_y_1) ** 2))

    # mse.append(test_mse)
plt.plot(np.arange(1, 101, 1), mse)
plt.show()
print(mse)

