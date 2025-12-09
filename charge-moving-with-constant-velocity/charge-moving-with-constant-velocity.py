
import sys
sys.path.append("..")

import numpy as np
import scipy as sp

from utils import *
from charge import *

class ChargeMovingWithConstantVelocity(Charge):

    def __init__(
        self,
        r0,
        v,
        e = sp.constants.elementary_charge,
        m = sp.constants.electron_mass
    ):
        super().__init__(e, m)
        self.r0 = r0
        self.v = v

    def position(self, t):
        return self.r0 + np.multiply.outer(t, self.v)

    def velocity(self, t):
        if (type(t) is np.ndarray):
            return np.tile(self.v, (t.shape[0], 1))
        else:
            return self.v

    def acceleration(self, t):
        if (type(t) is np.ndarray):
            return np.tile(np.zeros_like(self.v), (t.shape[0], 1))
        else:
            return np.zeros_like(self.v)
