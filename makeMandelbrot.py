#!/usr/bin/python

# Forked from:
# https://github.com/tobigithub/tensorflow-deep-learning/blob/master/examples/mandelbrot-tensorflow.py

import os

import numpy as np
import tensorflow as tf

from fractal import RunIterator, DisplayFractal

def MandelbrotIterativeFunction(Z):
    # Create c, z for grid Z
    c = tf.constant(Z.astype(np.complex64))
    z = tf.Variable(c)
    ns = tf.Variable(tf.zeros_like(c, tf.float32))

    # Initialize session variables
    tf.global_variables_initializer().run()

    # See: https://en.wikipedia.org/wiki/Mandelbrot_set#Formal_definition
    z_ = z * z + c
    not_diverged = tf.abs(z_) < 4

    step = tf.group(
        z.assign(z_),
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
    yl, yh = -1.3, 1.3
    xl, xh = -2, 1
    res = 0.005

    # Create session
    session = tf.InteractiveSession()

    # Create grid
    Y, X = np.mgrid[yl:yh:res, xl:xh:res]
    Z = X + Y * 1j

    # Get result of iterator and form it into an image
    ns = RunIterator(MandelbrotIterativeFunction, Z, steps)
    im = DisplayFractal(ns.eval())

    # Show fractal image
    im.show()
