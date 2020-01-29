import numpy as np

raw = np.genfromtxt(r"./airfoil_self_noise.dat",
                    dtype=None,
                    delimiter="\t")
airfoil = []
for t in raw:
    airfoil.append(list(t))

airfoil = np.array(airfoil)
x = airfoil[:, 0:5]
intercept = np.ones(len(airfoil)).reshape(len(airfoil), 1)
x = np.hstack((x, intercept))
y = airfoil[:, 5].reshape(len(airfoil), 1)

# regression
train_w = np.asarray([0, 0, 0, 0, 0, 0]).reshape(6, 1)
max_iter = 10000
learning_rate = 1e-15

for i in range(max_iter):
    gradient = np.dot(np.transpose(y - np.dot(x, train_w)), x)
    train_w = train_w - learning_rate * np.transpose(gradient)

solution = np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.dot(np.transpose(x), y))
print(train_w)
print(solution)
print(np.linalg.norm(train_w - solution)/np.linalg.norm(solution))