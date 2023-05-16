from euler import Euler
from krylov import Krylov
from laf import LinearAssignmentFlow

import numpy as np
from skimage import img_as_float
from skimage.io import imread
import matplotlib.pyplot as plt


# ------------------ setup ------------------

img = img_as_float(imread("../images/Scales1.png"))

# ground truth labels
labels = np.asarray(
    [[0.203, 0.656, 0.324],  # darkgreen
    [0.891, 0.043, 0.0],  # red
    [0.258, 0.188, 0.953],  # dark blue
    [0.98, 0.848, 0.02],  # yellow
    [0.082, 0.973, 0.375],  # lightgreen
    [0.258, 0.52, 0.953],  # light blue
    [0.902, 0.227, 0.867],  # violet
    [0.914, 0.484, 0.141]]  # brown
)

# LAF
laf = LinearAssignmentFlow(img, labels)
A, b = laf()
x0 = np.zeros(b.shape)


# ----------------- options -----------------

stepsizes = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
distances = [1, 2, 4, 8]

# ---------------- experiment ---------------

# ground truth
krylov = Krylov(A, b, img.shape)
gt_results = krylov(x0, 100, distances, labels)
gt_results = np.hstack(gt_results)

euler = Euler(A, b)
euler_results = []
for stepsize in stepsizes:
    results = euler(x0, stepsize, distances)
    euler_results.append(results)

euler_results = np.stack(euler_results)

error = np.linalg.norm(euler_results - gt_results, axis=1)

for i in range(len(distances)):
    plt.loglog(stepsizes, error[:, i], label=fr"$t$ = {distances[i]}")
    plt.scatter(stepsizes, error[:, i])
plt.xticks(stepsizes, rotation=35)
plt.xlabel("stepsize h")
plt.legend()
plt.show()


