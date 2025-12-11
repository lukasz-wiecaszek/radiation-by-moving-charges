
import sys
sys.path.append("..")

import numpy as np
import scipy as sp

from typing import Callable

from utils import *
from charge import *

class ChargeMovingWithConstantAngularVelocity(Charge):

    def __init__(
        self,
        trajectory,
        dt,
        e = sp.constants.elementary_charge,
        m = sp.constants.electron_mass
    ):
        super().__init__(e, m)
        self.trajectory = trajectory
        self.dt = dt

    def position(self, t):
        return self.trajectory(t)

    def velocity(self, t):
        delta = self.position(t + self.dt) - self.position(t)
        return delta / self.dt

    def acceleration(self, t):
        delta = self.velocity(t + self.dt) - self.velocity(t)
        return delta / self.dt
