import numpy as np

class NonlinearPendulum:
    def __init__(self, m, g, l):
        self.m = m
        self.g = g
        self.l = l

    def Tgrad(self):
        def grad(p): return p
        return grad

    def Vgrad(self):
        def grad(q): return self.m*self.g*self.l*np.sin(q)
        return grad
