# Fractals with Tensorflow

This extends the tensorflow tutorial to [create the Mandelbrot Set](https://www.tensorflow.org/tutorials/mandelbrot) to create any fractals created with iterative functions.

### Usage
You may change the configurations such as x, y range and the image resolution in the respective python files before running either `python3 makeJulia.py` or `python3 makeMandelbrot.py`.

For Julia sets, you may supply the value of the incremental constant `c` by running `python3 makeJulia.py x+yj` where `x` and `y` are floating point values. If no argument is supplied, then `c=-0.4+0.6j` by default. For instance, `python3 makeJulia.py 0.285+0.01j` is a valid argument.

### Requirements
* Tensorflow
* numpy
* Python Imaging Library
