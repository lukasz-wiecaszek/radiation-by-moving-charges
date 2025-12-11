import numpy as np

class Circle:

    # Define the basis vectors
    e0 = np.array([1, 0, 0])
    e1 = np.array([0, 1, 0])
    e2 = np.array([0, 0, 1])

    def __init__(self, r, f):
        self.r = r
        self.omega = 2 * np.pi * f

    def parametric_equation(self, t):
        return (np.multiply.outer(self.r * np.cos(self.omega * t), Circle.e0) +
                np.multiply.outer(self.r * np.sin(self.omega * t), Circle.e1))

    def max_linear_velocity(self):
        return self.omega * self.r
