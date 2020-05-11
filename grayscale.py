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
import math
import pycuda.driver as cuda
import pycuda.autoinit # automatically inits backend for you
from pycuda.compiler import SourceModule
import PIL.Image as imgur


class ImageFilter:

    src_module = """
    __global__ void grayscale_filter(unsigned char *red,
                                     unsigned char *green,
                                     unsigned char *blue,
                                     const unsigned int height,
                                     const unsigned int width) {
        const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
        const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
        if(row < height && col < width) {
            const unsigned int index = col + row * width;
            const unsigned char intensity = static_cast<unsigned char>(
                red[index] * 0.3 + green[index] * 0.59 + blue[index] * 0.11
            );
            red[index] = intensity;
            green[index] = intensity;
            blue[index] = intensity;
        }
    }
    """

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
    image_file = args['INPUT_IMAGE']
    # Open image and returns uchar4 array.
    uchar4_array = create_uchar4_array_from_image_file(image_file) # uchar4 automatically
    grayscale_array = ImageFilter(uchar4_array).grayscale
    imgur.fromarray(grayscale_array).save(args['OUPUT_IMAGE'])
    print(grayscale_array)


if __name__ == '__main__':
    main()
