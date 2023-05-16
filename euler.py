from time import time
import numpy as np
from tqdm import tqdm


class Euler:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """ Explicit Euler Integration
        :param A: ODE Matrix
        :param b: Source term of the ODE (Ax + b)
        """
        self.A = A
        self.b = b
        self.calls = []

    def __call__(self, x0: np.ndarray, dt: float, distances: float, *args, **kwargs):
        """ Forward Step
        :param x0: initial value
        :param dt: stepsize
        :param distances: List of distances that should be integrated up to
        :return:
        """

        x = x0
        distance = np.atleast_1d(distances)
        n_steps = np.ceil(distance / dt).astype(int)
        if isinstance(n_steps, np.ndarray):
            checkpoints = np.hstack((n_steps[0], np.diff(n_steps)))
        else:
            checkpoints = [n_steps]

        results = []
        times = []
        with tqdm(total=n_steps[-1]) as pbar:
            t0 = time()
            for n in checkpoints:
                for i in range(n):
                    x = x + (self.A @ x + self.b) * dt
                    pbar.update()
                results.append(x)
                times.append(time() - t0)

        results = np.hstack(results)
        call = {
            'x0': x0,
            'dt': dt,
            'distance': distance,
            'results': results,
            'times': np.asarray(times)
        }
        self.calls.append(call)

        return results