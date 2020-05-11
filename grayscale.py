#!/home/605/sincomb/anaconda3/bin/python3
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
import math # Built-in Math library
from time import time # Built-in time keeping library

from docopt import docopt # Easy Command Line I/O Library
import numpy as np # Linear alg library
import pycuda.driver as cuda # Access GPU specifics
import pycuda.autoinit # Automatically inits backend GPU stuffs for you
from pycuda.compiler import SourceModule # Complie cuda kernals
import PIL.Image as imgur # Amazing image processing library Pillow; OpenCV best for complex images


class ImageFilter:

    src_module = """
    __global__ void grayscale_filter(unsigned char *red,
                                     unsigned char *green,
                                     unsigned char *blue,
                                     const unsigned int height,
                                     const unsigned int width) {
        const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
        const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
        if ((row < height) && (col < width)) {
            const unsigned int index = col + row * width;
            const unsigned char intensity = static_cast<unsigned char>(
                red[index] * 0.3 + green[index] * 0.59 + blue[index] * 0.11
            );
            red[index] = intensity;
            green[index] = intensity;
            blue[index] = intensity;
        }
    }
    """ # Could have multiple __global__ kernals in here!

    def __init__(self, image_array, dim_block=32):
        self.module = SourceModule(self.src_module)
        self.image_array = image_array
        self.dim_block = dim_block

    @property
    def grayscale(self):

        red = self.image_array[:, :, 0].copy()
        green = self.image_array[:, :, 1].copy()
        blue = self.image_array[:, :, 2].copy()
        # self.image_array[:, :, 3] is the 4th pixel location reserved for opaqueness

        # (number of rows, number of columns, pixel vector size - here its 4 for rgba)
        height, width, pixel_dimension = self.image_array.shape

        dim_grid_x = math.ceil(width / self.dim_block)
        dim_grid_y = math.ceil(height / self.dim_block)

        max_grid_dim_x = pycuda.autoinit.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
        max_grid_dim_y = pycuda.autoinit.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)

        if (max_grid_dim_x * max_grid_dim_y) < (dim_grid_x * dim_grid_y):
            raise ValueError('ERROR :: Image demensions :: Grid exceeds max')

        grayscale_filter = self.module.get_function('grayscale_filter')

        grayscale_filter(
            cuda.InOut(red),
            cuda.InOut(green),
            cuda.InOut(blue),
            np.uint32(height),
            np.uint32(width),
            block=(self.dim_block, self.dim_block, 1),
            grid=(dim_grid_x, dim_grid_y)
        )

        grayscale_image_array = np.empty_like(self.image_array.copy())
        grayscale_image_array[:, :, 0] = red
        grayscale_image_array[:, :, 1] = green
        grayscale_image_array[:, :, 2] = blue

        return grayscale_image_array


def create_uchar4_array_from_image_file(image_file):
    """ Opens & Converts image to array of 4 unsigned 8-bit values """
    image = imgur.open(image_file) # Open image from path
    image_array = np.array(image) # Pulls uchar meta from image object
    return image_array


def main():
    args = docopt(__doc__)
    # Open image and returns uchar4 array.
    uchar4_array = create_uchar4_array_from_image_file(args['INPUT_IMAGE']) # uchar4 automatically
    time_start = time()
    # Returns altered image array with luminosity of -> (0.3 * R) + (0.59 * G) + (0.11 * B)
    grayscale_array = ImageFilter(uchar4_array).grayscale
    # Total time for GPU to have at it grayscaling the image.
    elapsed_time = time() - time_start
    # Create Pillow image object from new array.
    image = imgur.fromarray(grayscale_array)
    # JPG needs RGB by default
    image = image.convert('RGB')
    # Save to output path
    image.save(args['OUPUT_IMAGE'])
    # Prints elapsed seconds GPU took to convert to grayscale.
    print(f'{elapsed_time}')
    # DEBUG: print(grayscale_array)


if __name__ == '__main__':
    main()
