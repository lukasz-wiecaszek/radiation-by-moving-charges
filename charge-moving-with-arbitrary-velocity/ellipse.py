import numpy as np

class Ellipse:

    # Define the basis vectors
    e0 = np.array([1, 0, 0])
    e1 = np.array([0, 1, 0])
    e2 = np.array([0, 0, 1])

    # Define origin vector
    origin = np.array([-0.15e0, -0.15e0, 0])

    def __init__(self, a, b, f, theta):
        self.a = a
        self.b = b
        self.omega = 2 * np.pi * f
        self.rT = np.array([[+np.cos(theta), -np.sin(theta), 0],
                            [+np.sin(theta), +np.cos(theta), 0],
                            [0, 0, 1]]).T

    def __parametric_equation(self, t):
        return (np.multiply.outer(self.a * np.cos(self.omega * t), Ellipse.e0) +
                np.multiply.outer(self.b * np.sin(self.omega * t), Ellipse.e1))

    def parametric_equation(self, t):
        pos = self.__parametric_equation(t)
        return Ellipse.origin + np.dot(pos, self.rT)

    def min_linear_velocity(self):
        return self.omega * self.b

    def max_linear_velocity(self):
        return self.omega * self.a
