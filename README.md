rtHMM - Real-Time Hidden Markov Models
======================================

rtHMM is a header-only C++ library for real-time inference using Hidden Markov
Models. Although not finished yet, the API should be fairly stable by now.
The library provides inference algorithms for HMMs with continuous as well
as discrete, possibly multidimensional, observation distributions. A limited
number of probability distributions are already implemented, but it is easy
to create and use arbitrary distributions in this framework.

rtHMM can handle large state spaces, especially when they are sparsely
connected. It also allows for tying observation distributions to multiple
states and exploiting this fact during inference, which also improves
performance.

Note that the library is __NOT__ intended to provide learning functionality.
This should be done using better suited tools. If you need a HMM library for
__fast__ inference with an API designed with continuous real-time input in
mind, this one is for you.

## Build Info:

rtHMM is programmed in C++11. Building has only been tested using g++ 4.8.2
Ubuntu Linux 14.04, but any compiler supporting the necessary features should
be able to build it.

### Dependencies

rtHMM needs eigen3, and if you want to run unit tests the google test
framework:

http://eigen.tuxfamily.org/
http://code.google.com/p/googletest/

rtHMM uses CMake as build system. Google test can be either installed in a way
that CMake can find it, or placed in the 3rd_party directory unter "gtest".
A CMake script to find Eigen3 is provided.

It is planned to make Eigen3 only an optional dependency, but we're not there
yet.

### Build

To build rtHMM (on Linux, anything else has not been tested yet):

    $ cd rtHMM_dir
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

### Run Unit Tests:

    $ make test

or

    $ ./test/runUnitTests

## Documentation

All user-facing functions and classes are documented using a Doxygen-compatible
format. If you have doxygen installed, simply run

    $ doxygen rtHMM.doxygen.cfg

which will create a subfolder 'docs' where doxygen generates documentation in
both HTML and LaTeX files.

## License

rtHMM is licensed under the MIT/X Consortium License. See the file LICENSE
for further details.

## Future Work

Still missing:
 * Fixed-lag smoothing
 * File IO: Reading models from files
 * More tests to make sure everything is correct
 * Multithreading
