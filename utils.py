
import numpy as np
import scipy as sp

# types
Vector = np.ndarray[3, np.dtype[np.float64]]

# constants
pi = sp.constants.pi
e = sp.constants.elementary_charge
m = sp.constants.electron_mass
c = sp.constants.speed_of_light
eps0 = sp.constants.epsilon_0
mu0 = sp.constants.mu_0

# functions
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

# classes
class coords():
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def get(self):
        return (self.x, self.y)

    def inc(self):
        self.x += self.dx
        self.y += self.dy
