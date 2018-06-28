#!/usr/bin/python

# Forked from:
# https://github.com/tobigithub/tensorflow-deep-learning/blob/master/examples/mandelbrot-tensorflow.py

import os

import numpy as np
import tensorflow as tf

from fractal import RunIterator, DisplayFractal

def JuliaIterativeFunction(Z):
    # Set c, z, f for grid Z
    c = tf.constant(-0.4 + 0.6j)
    z = tf.constant(Z.astype(np.complex128))
    f = tf.Variable(z)
    ns = tf.Variable(tf.zeros_like(z, tf.float32))

    # Initialize session variables
    tf.global_variables_initializer().run()

    # See: https://en.wikipedia.org/wiki/Julia_set#Quadratic_polynomials
    f_ = f * f + c
    not_diverged = tf.abs(f_) < 2

    step = tf.group(
        f.assign(f_),
        ns.assign_add(tf.cast(not_diverged, "float32"))
    )

    # Return TF Group and NS
    return step, ns

if __name__ == '__main__':

    # Disable AVX/FMA warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Iteration Steps
    steps = 200

    # Range and resolution for grid
    yl, yh = -2, 2
    xl, xh = -2, 2
    res = 0.001

    # Create session
    session = tf.InteractiveSession()

    # Create grid
    Y, X = np.mgrid[yl:yh:res, xl:xh:res]
    Z = X + Y * 1j

    # Get result of iterator and form it into an image
    ns = RunIterator(JuliaIterativeFunction, Z, steps)
    im = DisplayFractal(ns.eval())

    # Show fractal image
    im.show()
