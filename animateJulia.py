#!/usr/bin/python

# Forked from:
# https://github.com/tobigithub/tensorflow-deep-learning/blob/master/examples/mandelbrot-tensorflow.py

import os

import numpy as np
import tensorflow as tf

import imageio
import cmath

from time import time

# import colors
from fractal import RunIterator, DisplayFractal
from makeJulia import setIncrement, JuliaIterativeFunction

if __name__ == '__main__':

    # Disable AVX/FMA warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Iteration Steps
    steps = 64

    # Range and resolution for grid
    yl, yh = -2, 2
    xl, xh = -2, 2
    res = 0.01

    # Create session
    session = tf.InteractiveSession()

    # Create grid
    Y, X = np.mgrid[yl:yh:res, xl:xh:res]
    Z = X + Y * 1j

    # Get result of iterator and form it into an image, which is sent to the writer
    # colors.SetColorProfile(colors.PURPLE)
    start = time()
    with imageio.get_writer('julia.gif', mode='I') as w:
        theta = 0
        frames = 10
        delta = 2 * cmath.pi/frames
        r = 0.7885
        for i in range(frames):
            increment = setIncrement(r, theta)
            theta += delta
            print("Frame %d:\n\tc = %s\n\t" % (i, str(increment)), end="")
            ns = RunIterator(JuliaIterativeFunction, Z, steps)
            im = DisplayFractal(ns.eval())
            w.append_data(np.array(im))
    print("\nOperation finished in:", time() - start, "seconds")
