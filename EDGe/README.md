# Model

This is an implementation of the fully convolutional encoder-decoder based generative (EDGe) model for power grid prediction

- Input: image-like multi-channel tensor, e.g. 4-channel input consisting column resistance, row resistance, current source and time.
- Output: single-channel IRDrop map with the same size of input.

## Installation

GridNet requires TensorFlow 1.x to be installed as backend. It was tested on Fermi server in Anaconda virtual env with following dependencies:

- python=3.8.5
- tensorflow=2.7
- numpy
- matplotlib