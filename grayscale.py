#!/home/605/sincomb/anaconda3/bin/python3
# ./grayscale.py  [--output_path=PATH] INPUT_IMAGE
""" Grayscale your image!

Usage:
    ./grayscale.py  (-h | --help)
    ./grayscale.py  INPUT_IMAGE
    ./grayscale.py  INPUT_IMAGE OUPUT_IMAGE

Arguments:
    INPUT_IMAGE
    OUPUT_IMAGE

Options:
    -h, --help                Prints out usage examples.
    -o, --output_path=PATH    Output Folder Path

Terminal Examples:

Import Example:

"""
from docopt import docopt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # automatically inits backend for you
from pycuda.compiler import SourceModule
import PIL.Image as imgur


def create_uchar4_array_from_image_file(image_file):
    """ Opens & Converts image to array of 4 unsigned 8-bit values """
    image = imgur.open(image_file) # Open image from path
    source_array = np.array(image) # Pulls uchar meta from image object
    return source_array


def grayscalify(uchar4_array):
    """  """
    return


def main():
    args = docopt(__doc__)
    image_file = args['INPUT_IMAGE']
    # Open image and returns uchar4 array.
    uchar4_array = create_uchar4_array_from_image_file(image_file) # uchar4 automatically
    # (number of rows, number of columns, pixel vector size - here its 4 for rgba)
    height, width, pixel_dimension = uchar4_array.shape
    print(uchar4_array)


if __name__ == '__main__':
    main()
