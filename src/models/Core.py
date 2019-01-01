import numpy as np
import matplotlib as mpl


# import tensorflow as tf


class AbstractModel:

    def __init__(self, X, time=0, dt=1e-4):
        self.dt = dt
        self.X = X
        self.time = time
        self.history = [[], []]

    def _step(self, I=None):
        pass

    def run(self, start, end, X_0=None):
        self.time = start
        if X_0 is not None:
            self.X = X_0
        for t in range(int((end - start + 1) / self.dt)):
            self.history[0].append(t * self.dt)
            self.history[1].append(self.X)
            self._step()
        return self


class Pole(AbstractModel):

    def __init__(self, angle, angular_velocity, displacement, velocity, dt=1e-4, Mass=10, mass=1, length=1, time=0):
        AbstractModel.__init__(self, X=np.array([displacement, angle, velocity, angular_velocity]), time=time, dt=dt)
        self.M = Mass
        self.m = mass
        self.l = length

    def _f(self):
        return np.linalg.inv(
            np.array([[-np.cos(self.X[1]), self.l], [self.M + self.m, self.m * self.l * np.cos(self.X[1])]]))

    def _g(self, F):
        return np.array([np.sin(self.X[1]), F - self.m * self.l * np.square(self.X[2]) * np.sin(self.X[1])])

    def _step(self, F=None):
        self.X = self.X + self.rhs(F) * self.dt         #be careful += does not work
        self.time += self.dt
        return self

    def rhs(self, I):
        if I is None:
            I = 0
        return np.concatenate([self.X[2:], np.matmul(self._f(), self._g(I))])


#class 