from time import time
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from scipy.linalg import expm
from typing import Tuple, List


class Krylov:
    def __init__(self, A: np.ndarray, b: np.ndarray, img_dim: Tuple[int]):
        """ Krylov Subspace Integration
        :param A: ODE Matrix
        :param b: Source term of the ODE (Ax + b)
        """
        self.A = A
        self.b = b
        self.img_dim = img_dim
        self.calls = []

    def _lanczos(self, m: int):
        """Lanczos iteration, optimized for symmetric matrices
        :param m: Krylov subspace dimension
        """
        V = np.zeros((len(self.b), m))  # len + assert
        H = np.zeros((m, m))
        v = self.b / np.linalg.norm(self.b)
        V[:, [0]] = v
        w = self.A @ v
        t_o = v.T @ w  # t_o -> diagonal entries, t_od -> offdiagonal
        H[[0], [0]] = t_o
        w = w - t_o * v
        for j in tqdm(range(1, m)):
            t_od = np.linalg.norm(w)
            H[j, j - 1] = H[j - 1, j] = t_od
            v = w / t_od
            V[:, [j]] = v
            w = self.A @ v
            t_o = v.T @ w
            H[j, j] = t_o
            w = w - t_o * v - t_od * V[:, [j - 1]]
        h_err = np.linalg.norm(w)
        v_err = w / h_err
        return V, H, h_err, v_err

    def _augment_H(self, H: np.ndarray, p: int):
        """Augments H such that we can easily compute phi_q(H) for all q <= p"""
        m = H.shape[0]
        H_ = np.zeros((m + p, m + p))
        H_[:m, :m] = H
        H_[0, m] = 1
        for k in range(p - 1):
            H_[m + k, m + 1 + k] = 1
        return H_

    def __call__(self, x0: np.ndarray, m: int, distances: List[float], err_correction: bool = True, err_approx: int = 0,
                 save_calls: bool = True, labels_out: bool = False):
        """ Forward Step
        :param x0: initial value
        :param m: Krylov subspace dimension
        :param distances: List of distances that should be integrated up to
        :param err_correction: should output be corrected by first error term
        :param err_approx: number of error terms to approximate up to (0...)
        :param save_calls: save parameters, results, ... in a dict for every call
        :param labels_out: only return indices of the largest valued labels
        :return: List of flattened images for all distances
        """
        t0 = time()
        V, H, h_err, v_err = self._lanczos(m)  # orthonormalization

        results = []
        approx_list = []
        err_approx += err_correction  # increase by one if true
        for t in distances:  # start a new run for every distance
            p = err_approx + 1  # basic -> p=1, corrected -> p=2, +1 for every error approximation
            Ht = H * t  # evaluate at time t
            Ht_ = self._augment_H(Ht, p)
            phi_H = expm(Ht_)
            phi_e = phi_H[0:m, [-p]]  # last col yields phi_p(H)*e_1

            beta = np.linalg.norm(self.b)
            img = t * (V @ phi_e) * beta
            if np.count_nonzero(x0) != 0:
                phi_0_e = phi_H[0:m, [1]]
                img += (V @ phi_0_e) * np.linalg.norm(x0)
            if err_correction:  # correct image
                phi = phi_H[m-1, -err_approx]
                err = t * t * beta * h_err * phi * v_err
                img += err

            approx = None
            if err_approx > 0:
                if not err_correction:  # in first step no additional A is needed
                    phi = phi_H[m-1, -err_approx]
                    error = beta * h_err * t * t * phi * v_err
                    approx = [error.copy()]
                else:  # an additional A multiplication is needed -> do that in loop
                    error = np.zeros_like(v_err)
                    approx = []
                if err_approx > 1:
                    A_ = self.A
                    for i in range(3, 2 + err_approx):
                        phi = phi_H[m-1, -err_approx + i - 2]
                        error += beta * h_err * (t ** i) * phi * (A_ @ v_err)
                        approx.append(error.copy())
                        if i != (err_approx + 1): # not needed in last step
                            A_ = A_ @ self.A
                    approx = np.linalg.norm(approx, axis=1).squeeze()
            # img = img.reshape(self.img_dim[:2] + (-1,))
            if labels_out:
                img = np.argmax(img, axis=2)
            results.append(img)
            if save_calls:
                approx_list.append(approx)

        if save_calls:
            call = {
                'x0': x0,
                'm': m,
                'distances': distances,
                'results': results,
                'approx': approx_list,
                'times': np.asarray([time() - t0])
            }
            self.calls.append(call)

        return results
