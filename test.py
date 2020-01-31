import numpy as np

true_w = [4, 15, 20, 10, 5, 15]
X = np.random.randn(100, 5)  # m independent vectors of n variables
intercept = np.ones(100).reshape(100, 1)
X = np.hstack((X, intercept))
y = np.dot(X, true_w)  # response variable

train_w = np.zeros_like(true_w)

max_iter = 1000
learning_rate = 0.01

for i in range(max_iter):
    gradient = (np.dot(np.dot(np.transpose(X), X), train_w) - np.dot(np.transpose(X), y))
    train_w = train_w - learning_rate * np.transpose(gradient)

print(train_w)
print(true_w)
print(y.shape)

a = np.asarray(np.arange(20)).reshape(4,5)
print(a)

print(a[0:3, :])