def naive_addition(x, y):
    """ Adds two matrices """
    assert len(x.shape) == 2, "Shape of x or y not rank 2!"
    assert x.shape == y.shape, "Shape of x and y not the same!"

    x = x.copy()
    y = y.copy()

    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            x[i, j] += y[i, j]

    return x


import numpy as np

x = np.array([[1, 2, 3], [4, 5, 7], [2, 3, 1]]).astype(float)
y = np.array([[1, 2, 3], [4, 5, 7], [3, 23.4, 3]]).astype(float)

print(naive_addition(x, y))
