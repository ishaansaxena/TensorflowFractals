#!/usr/bin/python

# Forked from:
# https://github.com/tobigithub/tensorflow-deep-learning/blob/master/examples/mandelbrot-tensorflow.py

import os
import sys

import numpy as np
import tensorflow as tf

# import colors
from fractal import RunIterator, DisplayFractal

C = -0.4 + 0.6j

def JuliaIterativeFunction(Z):
    # Set c, z, f for grid Z
    c = tf.constant(C)
    z = tf.constant(Z.astype(np.complex128))
    f = tf.Variable(z)
    ns = tf.Variable(tf.zeros_like(z, tf.float32))

    # Initialize session variables
    tf.global_variables_initializer().run()

    # See: https://en.wikipedia.org/wiki/Julia_set#Quadratic_polynomials
    f_ = f * f + c
    not_diverged = tf.abs(f_) < 2

    # TODO: Improve incremental function complexity
    step = tf.group(
        f.assign(f_),
        ns.assign_add(tf.cast(not_diverged, "float32"))
    )

    # Return TF Group and NS
    return step, ns

if __name__ == '__main__':

    # Disable AVX/FMA warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Set Constant value if present
    try:
        increment = complex(sys.argv[1])
        C = increment
    except:
        pass

    # Iteration Steps
    steps = 64

    # Range and resolution for grid
    yl, yh = -2, 2
    xl, xh = -2, 2
    res = 0.005

    # Create session
    session = tf.InteractiveSession()

    # Create grid
    Y, X = np.mgrid[yl:yh:res, xl:xh:res]
    Z = X + Y * 1j

    # Get result of iterator and form it into an image
    # colors.SetColorProfile(colors.PURPLE)
    ns = RunIterator(JuliaIterativeFunction, Z, steps)
    im = DisplayFractal(ns.eval())

    # Show fractal image
    im.show()
