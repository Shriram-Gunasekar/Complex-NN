import numpy as np

class Octonion:
    def __init__(self, real, e1, e2, e3, e4, e5, e6, e7):
        self.real = real
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.e4 = e4
        self.e5 = e5
        self.e6 = e6
        self.e7 = e7

    def __repr__(self):
        return f"({self.real}, {self.e1}, {self.e2}, {self.e3}, {self.e4}, {self.e5}, {self.e6}, {self.e7})"

    def __mul__(self, other):
        real = self.real * other.real - self.e1 * other.e1 - self.e2 * other.e2 - self.e3 * other.e3 \
               - self.e4 * other.e4 - self.e5 * other.e5 - self.e6 * other.e6 - self.e7 * other.e7
        e1 = self.real * other.e1 + self.e1 * other.real + self.e2 * other.e7 - self.e3 * other.e6 \
             + self.e4 * other.e5 - self.e5 * other.e4 + self.e6 * other.e3 - self.e7 * other.e2
        e2 = self.real * other.e2 + self.e2 * other.real + self.e3 * other.e1 - self.e1 * other.e3 \
             + self.e4 * other.e6 - self.e6 * other.e4 + self.e5 * other.e7 - self.e7 * other.e5
        e3 = self.real * other.e3 + self.e3 * other.real + self.e1 * other.e2 - self.e2 * other.e1 \
             + self.e4 * other.e7 - self.e7 * other.e4 + self.e6 * other.e5 - self.e5 * other.e6
        e4 = self.real * other.e4 + self.e4 * other.real + self.e1 * other.e5 - self.e5 * other.e1 \
             + self.e2 * other.e6 - self.e6 * other.e2 + self.e3 * other.e7 - self.e7 * other.e3
        e5 = self.real * other.e5 + self.e5 * other.real + self.e1 * other.e4 - self.e4 * other.e1 \
             + self.e2 * other.e7 - self.e7 * other.e2 + self.e6 * other.e3 - self.e3 * other.e6
        e6 = self.real * other.e6 + self.e6 * other.real + self.e1 * other.e7 - self.e7 * other.e1 \
             + self.e3 * other.e4 - self.e4 * other.e3 + self.e5 * other.e2 - self.e2 * other.e5
        e7 = self.real * other.e7 + self.e7 * other.real + self.e1 * other.e6 - self.e6 * other.e1 \
             + self.e2 * other.e5 - self.e5 * other.e2 + self.e4 * other.e3 - self.e3 * other.e4
        return Octonion(real, e1, e2, e3, e4, e5, e6, e7)

def random_octonion():
    real = np.random.randn()
    e1 = np.random.randn()
    e2 = np.random.randn()
    e3 = np.random.randn()
    e4 = np.random.randn()
    e5 = np.random.randn()
    e6 = np.random.randn()
    e7 = np.random.randn()
    return Octonion(real, e1, e2, e3, e4, e5, e6, e7)
