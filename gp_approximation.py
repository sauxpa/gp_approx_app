import numpy as np
from functools import partial

class Kernel():
    def __init__(self,
                 data=None,
                 y=None,
                 scale=1.0,
                 noise=1.0,
                 kernel_name='gauss',
                 ):
        self.data = data
        self.y = y
        self.scale = scale
        self.noise = noise
        self.kernel_name = kernel_name

    @property
    def n(self):
        return len(self.data)

    def __call__(self, x, y):
        if self.kernel_name == 'gauss':
            return np.exp(-np.abs(x-y)**2/(2*self.scale**2))
        elif self.kernel_name == 'exp':
            return np.exp(-np.abs(x-y)/self.scale)
        else:
            raise Exception('{} not implemented!'.format(self.kernel_name))

    def kernel_matrix(self):
        i, j = np.indices((self.n, self.n))
        g = np.vectorize(self.__call__, otypes=[np.ndarray])
        return g(self.data[i], self.data[j]).astype(np.float64)

    def weights(self):
        K = self.kernel_matrix() + self.noise**2 * np.eye(self.n)
        return np.linalg.solve(K, self.y)

    def basis(self):
        self.basis = []
        for x in self.data:
            self.basis.append(partial(self.__call__, x))
        return self.basis
