#!/usr/bin/env python

"""
This script tries to discover the secret behind the Winograd algorithm.
"""

import numpy as np

def FIR(x, b):
    """
    Standard FIR algorithm implementation.

    :param x: an input array
    :param b: a coefficient array of length r
    :return: FIR result
    """
    r = len(b)
    y = np.zeros(len(x) - r + 1)
    for i in range(len(y)):
        y[i] = np.sum(x[i:i+r] * b) 
    return y

def winograd_FIR(x, b, p = 2):
    """
    Winograd FIR algorithm implementation.

    :param x: an input array
    :param b: a coefficient array of length r
    :param p: Winograd matrix, number of rows
    :return: FIR result
    """

    r = len(b)
    n = len(x)
    y = np.zeros(n - r + 1)
    m = np.zeros(r + p - 1)
    
    assert p == 2, "only support p = 2"
    assert len(y) % p == 0, "len(y) should be divisible by p"

    num_tiles = int(len(y) / p)

    for t in range(num_tiles):
        i = t * p
        d = x[i:i+len(m)]

        m[0] = (d[0] - d[2]) * b[0]
        m[1] = (d[1] + d[2]) * (b[0] + b[1] + b[2]) / 2
        m[2] = (d[2] - d[1]) * (b[0] - b[1] + b[2]) / 2
        m[3] = (d[1] - d[3]) * b[2]

        y[i] = m[0] + m[1] + m[2]
        y[i + 1] = m[1] - m[2] - m[3]

    return y

def winograd_FIR_mm(x, b, p = 2):
    """
    Winograd FIR implemented by matrix multiplication.

    :param x: an input array
    :param b: a coefficient array of length r
    :param p: Winograd matrix, number of rows
    :return: FIR result
    """

    B = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]).T
    G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]])
    A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]]).T
    
    r = len(b)
    n = len(x)
    y = np.zeros(n - r + 1)
    m = np.zeros(r + p - 1)
    
    assert p == 2, "only support p = 2"
    assert len(y) % p == 0, "len(y) should be divisible by p"

    num_tiles = int(len(y) / p)

    for t in range(num_tiles):
        i = t * p
        d = x[i:i+len(m)]
        g = b

        y[i:i+2] = np.dot(A.T, np.dot(G, g) * np.dot(B.T, d))

    return y



if __name__ == '__main__':
    n = 16
    r = 3
    x = np.random.random(n)
    b = np.random.random(r)

    assert np.isclose(FIR(x, b), winograd_FIR(x, b)).all(), \
           "FIR and winograd should match"
    assert np.isclose(FIR(x, b), winograd_FIR_mm(x, b)).all(), \
           "FIR and winograd (MM) should match"

    print("FIR PASSED")
