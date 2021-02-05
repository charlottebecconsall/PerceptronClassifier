from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
import gzip, os


def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels). Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the assignment.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.

    Hint: look at the writeup for sample code on using the gzip library

    :param inputs_file_path: file path for inputs, something like
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """

    input_header_num = 16 #bytes
    label_header_num = 8 #bytes

    # TODO: Load inputs and labels
    with gzip.open(inputs_file_path) as inputs_file:
      inputs_file.read(input_header_num)
      inputs = inputs_file.read()
      input_array = np.frombuffer(inputs, dtype=np.uint8, count=(num_examples*784))

    with gzip.open(labels_file_path) as labels_file:
      labels_file.read(label_header_num)
      labels = labels_file.read()
      label_array = np.frombuffer(labels, dtype=np.uint8, count=num_examples)

    # TODO: Normalise inputs
    input_array = np.float32(input_array)
    input_array = input_array / 255

    # return inputs and labels array
    return (label_array, input_array)


def main():
  """Testing the preprocessing"""
  inputs_file_path = "/content/gdrive/My Drive/Colab Notebooks/Assignment 1/MNIST_data/t10k-images-idx3-ubyte.gz"
  labels_file_path = "/content/gdrive/My Drive/Colab Notebooks/Assignment 1/MNIST_data/t10k-labels-idx1-ubyte.gz"
  num_examples = 20
  image_size = 784
  arrays = get_data(inputs_file_path, labels_file_path, num_examples)
  label_array = arrays[0]
  input_array = arrays[1]
  print("LABELS")
  print(label_array)
