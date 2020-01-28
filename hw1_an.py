import numpy as np

raw = np.genfromtxt(r"./airfoil_self_noise.dat",
                    dtype=None,
                    delimiter="\t")
airfoil = []
for t in raw:
    airfoil.append(list(t))

airfoil = np.array(airfoil)
x = airfoil[:, 0:5]
y = airfoil[:, 5]
print(x)
print(y)

