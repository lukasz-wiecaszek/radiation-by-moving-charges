
import sys
sys.path.append("..")

import numpy as np
import scipy as sp

from utils import *
from charge import *

class OscillatingCharge(Charge):

    # Define the basis vectors
    e0 = np.array([1, 0, 0])
    e1 = np.array([0, 1, 0])
    e2 = np.array([0, 0, 1])

    def __init__(
        self,
        r0,
        amplitude,
        frequency,
        e = sp.constants.elementary_charge,
        m = sp.constants.electron_mass
    ):
        super().__init__(e, m)
        self.r0 = r0
        self.amplitude = amplitude
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency

    def position(self, t):
        return self.r0 + np.multiply.outer(self.amplitude * np.sin(self.omega * t), OscillatingCharge.e0)

    def velocity(self, t):
        return self.r0 + np.multiply.outer(self.amplitude * self.omega * np.cos(self.omega * t), OscillatingCharge.e0)

    def acceleration(self, t):
        return self.r0 + np.multiply.outer(-self.amplitude * self.omega**2 * np.sin(self.omega * t), OscillatingCharge.e0)
