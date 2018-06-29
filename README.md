# Fractals with Tensorflow

This extends the tensorflow tutorial to [create the Mandelbrot Set](https://www.tensorflow.org/tutorials/mandelbrot) to create any fractals which may be created with iterative functions.

### Usage
See [this gist](https://gist.github.com/ishaansaxena/0edd14c73aa7a5e8c6609af7d5f509a1) for information on usage and results.

For Julia sets, you may supply the value of the incremental constant `c` by running `python3 makeJulia.py x+yj` where `x` and `y` are floating point values. If no argument is supplied, then `c=-0.4+0.6j` by default. For instance, `python3 makeJulia.py 0.285+0.01j` is a valid argument.

### Requirements
* Tensorflow
* numpy
* Python Imaging Library
