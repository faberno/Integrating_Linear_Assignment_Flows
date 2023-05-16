import numpy as np
import scipy.sparse as sp

class LinearAssignmentFlow:
    def __init__(self, img: np.ndarray, prototypes: np.ndarray):
        """Implementation of the Linear Assignment Flow by Zeilmann et al.
        :param img: Image of shape (x, y, 3)
        :param prototypes: Labels of shape (z, 3)
        """
        self.img = img
        self.shape = img.shape
        self.prototypes = prototypes
        self.A = None
        self.b = None

    def _produceB(self):
        """Produces the source term b in Ax+b. Contains the image data."""
        D = self.img.reshape((-1, 3, 1))
        D = D - self.prototypes.T
        D = - np.sqrt((D ** 2).sum(axis=1))
        return (D - D.mean()).reshape(-1, 1)

    def __produceA(self):
        """Produces the ODE Matrix A in Ax+b. Represents the Pixels neighborhood."""
        i1 = self.shape[0]
        i2 = self.shape[1]

        j = len(self.prototypes)

        # la.circulant works too but slower for bigger values
        offdi = sp.eye(i1) + sp.eye(i1, k=-1) + sp.eye(i1, k=1) + \
                sp.eye(i1, k=i1 - 1) + sp.eye(i1, k=-(i1 - 1))
        offdi2 = sp.eye(i2) + sp.eye(i2, k=-1) + sp.eye(i2, k=1) + \
                 sp.eye(i2, k=i2 - 1) + sp.eye(i2, k=-(i2 - 1))

        A = (1 / 9) * sp.kron(offdi, offdi2)
        I = sp.eye(j, j)
        A = sp.kron(A, I)

        return A

    def __call__(self):
        """Compute LAF components for specified parameters
        :return: ODE Matrix (A) and source term (b)
        """
        b = self._produceB()
        A = self.__produceA()
        return A, b