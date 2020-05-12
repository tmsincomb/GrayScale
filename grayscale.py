#!/home/605/sincomb/anaconda3/bin/python3
""" Grayscale your image!

Usage:
    ./grayscale.py  (-h | --help)
    ./grayscale.py  INPUT_IMAGE
    ./grayscale.py  [--DIM_BLOCK=<numeric_value>] INPUT_IMAGE OUPUT_IMAGE

Arguments:
    INPUT_IMAGE                        Input GrayScaled JPG
    OUPUT_IMAGE                        Output GrayScaled JPG

Options:
    -h, --help                         Prints out usage examples.
    -d, --DIM_BLOCK=<numeric_value>    Output Folder Path [default: 32]

Terminal Examples:
    ./grayscale.py example_image.jpg out.jpg
    ./grayscale.py --DIM_BLOCK 32 example_image.jpg out.jpg
    ./grayscale.py -d 32 example_image.jpg out.jpg
"""
import math # Built-in Math library
from time import time # Built-in time keeping library

from docopt import docopt # Easy Command Line I/O Library
import numpy as np # Linear alg library
import pycuda.driver as cuda # Access GPU specifics
import pycuda.autoinit # Automatically inits backend GPU stuffs for you
from pycuda.compiler import SourceModule # Complie cuda kernels
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

        // If boundary hit don't keep going (i.e. grid * block exceed max)
        if ((row < height) || (col < width)) return;

        const unsigned int index = col + row * width;
        const unsigned char intensity = static_cast<unsigned char>(
            red[index] * 0.3 + green[index] * 0.59 + blue[index] * 0.11
        );
        red[index] = intensity;
        green[index] = intensity;
        blue[index] = intensity;
    }
    """ # Could have multiple __global__ kernels in here!

    def __init__(self, image_array, dim_block=32):
        self.module = SourceModule(self.src_module)
        self.image_array = image_array
        self.dim_block = dim_block

    @property
    def grayscale(self):
        """ Convert Image to Grayscale: luminosity of -> (0.3 * R) + (0.59 * G) + (0.11 * B) """
        # Copy dimension as to not write over address for future call.
        red, green, blue = np.copy(self.image_array[:, :, :])

        # number of rows, number of columns, and pixel vector size - here its 4 for rgba
        height, width, pixel_dimension = self.image_array.shape
        print(height)
        print(width)

        # Adjust grid to specified block
        dim_grid_x = math.ceil(width / self.dim_block)
        dim_grid_y = math.ceil(height / self.dim_block)
        print(dim_grid_x)
        print(dim_grid_y)

        # Determine max grid
        max_grid_dim_x = pycuda.autoinit.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
        max_grid_dim_y = pycuda.autoinit.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)
        print(max_grid_dim_x)
        print(max_grid_dim_y)

        # If grid determined from block size exceeds max grid, we have issues.
        if (max_grid_dim_x * max_grid_dim_y) < (dim_grid_x * dim_grid_y):
            raise ValueError('ERROR :: Image demensions :: Grid exceeds max')
        # Call specific function from CUDA kernel
        grayscale_filter = self.module.get_function('grayscale_filter')
        # Use grayscale function is specific grid, block, and array for color channels
        grayscale_filter(
            cuda.InOut(red),
            cuda.InOut(green),
            cuda.InOut(blue),
            np.uint32(height),
            np.uint32(width),
            block=(self.dim_block, self.dim_block, 1),
            grid=(dim_grid_x, dim_grid_y)
        )
        # Allocates array and it will not take the time to set the element values.
        grayscale_image_array = np.array([red, green, blue])
        # grayscale_image_array = np.empty_like(self.image_array.copy())
        # grayscale_image_array[:, :, 0] = red
        # grayscale_image_array[:, :, 1] = green
        # grayscale_image_array[:, :, 2] = blue

        return grayscale_image_array


def create_uchar4_array_from_image_file(image_file):
    """ Opens & Converts image to array of 4 unsigned 8-bit values """
    image = imgur.open(image_file) # Open image from path
    image = image.convert('RGB') # insure JPG format output
    image_array = np.array(image) # Pulls uchar meta from image object
    return image_array


def main():
    args = docopt(__doc__) # grab command inputs into a dictionary
    input_image = args['INPUT_IMAGE'] # image source input path
    dim_block = int(args['--DIM_BLOCK']) # default is 32; needs to be multiple of 32
    output_image = args['OUPUT_IMAGE'] # image output path
    # Open image and returns uchar4 array.
    uchar4_array = create_uchar4_array_from_image_file(input_image) # uchar4 automatically
    # Start GPU timer!
    time_start = time() # default is that it is in seconds
    # Returns altered image array with luminosity of -> (0.3 * R) + (0.59 * G) + (0.11 * B)
    grayscale_array = ImageFilter(image_array=uchar4_array, dim_block=dim_block).grayscale
    # Total time for GPU to have at it grayscaling the image.
    elapsed_time = time() - time_start
    # Create Pillow image object from new array.
    image = imgur.fromarray(grayscale_array)
    # Save to output path
    image.save(output_image)
    # Prints elapsed seconds GPU took to convert to grayscale.
    print(f'{round(elapsed_time, 5)}')
    # DEBUG: print(grayscale_array)


if __name__ == '__main__':
    main()
