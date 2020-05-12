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
    -d, --DIM_BLOCK=<numeric_value>    Block size for GPU squared [default: 16]

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
                                     unsigned int height,
                                     unsigned int width) {
        unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int index = col + row * width;

        // If boundary hit don't keep going (i.e. grid * block exceed max)
        if ((row > height) || (col > width)) return;

        // luminosity method
        unsigned char intensity = static_cast<unsigned char>(
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
        self.block_size = dim_block**2 # square block used i.e. (16, 16, 1)
        self.grid_size = None # last filter grid used
        # number of rows, number of columns, and pixel vector size - here its 4 for rgba
        self.height, self.width, self.pixel_dimension = self.image_array.shape
        self.image_size = self.height * self.width

    @property
    def grayscale(self):
        """ Convert Image to Grayscale: luminosity of -> (0.3 * R) + (0.59 * G) + (0.11 * B) """

        # Copy dimension as to not write over address for future call.
        red = np.copy(self.image_array[:, :, 0])
        green = np.copy(self.image_array[:, :, 1])
        blue = np.copy(self.image_array[:, :, 2])

        # Adjust grid to specified block size
        dim_grid_x = math.ceil(self.width / self.dim_block)
        dim_grid_y = math.ceil(self.height / self.dim_block)

        # Determine max grid
        max_grid_dim_x = pycuda.autoinit.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
        max_grid_dim_y = pycuda.autoinit.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)

        # If grid determined from block size exceeds max grid, we have issues.
        if (max_grid_dim_x * max_grid_dim_y) < (dim_grid_x * dim_grid_y):
            raise ValueError('ERROR :: Image demensions :: Grid exceeds max')

        # for easy sanity check tracking
        self.grid_size = dim_grid_x * dim_grid_y

        # Call specific function from CUDA kernel
        grayscale_filter = self.module.get_function('grayscale_filter')

        # Use grayscale function is specific grid, block, and array for color channels
        grayscale_filter(
            cuda.InOut(red),
            cuda.InOut(green),
            cuda.InOut(blue),
            np.uint32(self.height),
            np.uint32(self.width),
            block=(self.dim_block, self.dim_block, 1),
            grid=(dim_grid_x, dim_grid_y)
        )

        # Allocates array and it will not take the time to set the element values.
        grayscale_image_array = np.empty_like(self.image_array.copy())
        grayscale_image_array[:, :, 0] = red
        grayscale_image_array[:, :, 1] = green
        grayscale_image_array[:, :, 2] = blue

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
    height, width = uchar4_array.shape[:2]
    image_size = height * width

    # Start GPU timer!
    time_start = time() # default is that it is in seconds
    # Returns altered image array with luminosity of -> (0.3 * R) + (0.59 * G) + (0.11 * B)
    image_filter = ImageFilter(image_array=uchar4_array, dim_block=dim_block)
    grayscale_array = image_filter.grayscale
    # Total time for GPU to have at it grayscaling the image.
    elapsed_time = round((time() - time_start), 5)

    # Create Pillow image object from new array.
    image = imgur.fromarray(grayscale_array)
    # Save to output path
    image.save(output_image)
    # Prints elapsed seconds GPU took to convert to grayscale.
    # image_size | block_size | grid_size | time (seconds)
    print(f'{image_filter.image_size}\t{image_filter.block_size}\t{image_filter.grid_size}\t{elapsed_time}')
    # DEBUG: print(grayscale_array)


if __name__ == '__main__':
    main()
