import numpy as np


nx = 8
ny = 8
nz = 8
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)

X, Y, Z = np.meshgrid(x, y, z)
np.savetxt(f"x_{nx}_{ny}_{nz}.txt", X.flatten())
