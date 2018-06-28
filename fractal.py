#!/usr/bin/python

# Forked from:
# https://github.com/tobigithub/tensorflow-deep-learning/blob/master/examples/mandelbrot-tensorflow.py

import math
import numpy as np

from time import time
from PIL import Image

def DisplayFractal(a):
    a_cyclic = ((2 * math.pi / 20) * a).reshape(list(a.shape) + [1])
    # Apply colors
    img = np.concatenate([
        80 - 50 * np.cos(a_cyclic),
        0 + 1 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    im = Image.fromarray(a)
    return im

def RunIterator(function, Z, steps):
    # Get MandelbrotIterativeFunction
    step, ns = function(Z)

    start = time()
    for i in range(steps): step.run()
    print("Finished in", time() - start, "seconds")

    return ns
