
import numpy as np
import scipy as sp
from abc import ABC, abstractmethod

from utils import *

class Charge(ABC):
    """Abstract class representing a point charge"""
    def __init__(
        self,
        e: float = sp.constants.elementary_charge,
        m: float = sp.constants.electron_mass
    ):
        self.e = e
        self.m = m

    @abstractmethod
    def position(self, t):
        """
        Function returns position(s) of the charge at time(s) 't'.
        If 't' is a list, a list of positions is returned,
        where each element corresponds to the position at time t.

        Parameters
        ----------
        t : float | np.ndarray[np.dtype[np.float64]]
            Time(s) for which the position(s) is(are) to be calculated.

        Returns
        -------
        Charge's position(s) corresponding to time(s) 't'.
        """
        pass

    @abstractmethod
    def velocity(self, t):
        """
        Function returns velocity(ies) of the charge at time(s) 't'.
        If 't' is a list, a list of velocities is returned,
        where each element corresponds to the velocity at time t.

        Parameters
        ----------
        t : float | np.ndarray[np.dtype[np.float64]]
            Time(s) for which the velocity(ies) is(are) to be calculated.

        Returns
        -------
        Charge's velocity(ies) corresponding to time(s) 't'.
        """
        pass

    @abstractmethod
    def acceleration(self, t):
        """
        Function returns acceleration(s) of the charge at time(s) 't'.
        If 't' is a list, a list of accelerations is returned,
        where each element corresponds to the acceleration at time t.

        Parameters
        ----------
        t : float | np.ndarray[np.dtype[np.float64]]
            Time(s) for which the acceleration(s) is(are) to be calculated.

        Returns
        -------
        Charge's acceleration(s) corresponding to time(s) 't'.
        """
        pass

    def distance(self, r, t):
        """
        Function returns distance between the charge at the position(s)
        at the retarded time(s) 't' to the point(s) of observation 'r'.

        Parameters
        ----------
        r : Vector | np.ndarray[np.dtype[Vector]]
            Point(s) of observation.
        t : float | np.ndarray[np.dtype[np.float64]]
            Retarded time(s) of the charge corresponding to the point(s) of observation.

        Returns
        -------
        Distance of the charge at the position(s) at the retarded time(s) 't'
        to the point(s) of observation 'r'.
        """
        diff = r - self.position(t)
        return np.linalg.norm(diff, axis=-1)
