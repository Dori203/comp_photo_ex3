import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from skimage.color import rgb2xyz, xyz2rgb
from scipy.ndimage.filters import convolve
from skimage.draw import line
from skimage.color import rgb2gray
import random
import tensorflow as tf
import os
import imageio
import cv2


# Constants
GRAY = 1
RGB = 2
MAX_INTENSITY = 255
BASIC_GAUSS = np.array([1, 1])
MIN_SIZE = 32 # fix number 1. changed from 16 to 32.

def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size-1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2*half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size-1 - p1[0], kernel_size-1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1-norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2*half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size-1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel

def random_motion_blur(image, list_of_kernel_sizes):
    """

    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return:
    """
    # sample an angle and kernel size uniformaly.
    alpha = random.uniform(0, np.pi)
    index = random.randrange(0, len(list_of_kernel_sizes))
    # apply chosen values to image.
    conv_image = add_motion_blur(image, list_of_kernel_sizes[index], alpha)
    # round results.
    conv_image *= 255
    conv_image = np.rint(conv_image)
    conv_image /= 255

    # clip results to [0,1]
    result = np.clip(conv_image, 0, 1)
    return result

def add_motion_blur_1(image, kernel_size, angle):
    """
    :param image: a RGB image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return:
    """
    # get kernel
    kernel = motion_blur_kernel(kernel_size, angle)

    # convolve image.
    for i in range(3):
        image[:,:,i] = convolve(image[:,:,i], kernel, mode='nearest')
    return image

def read_image(filename, representation):
    """
    Reads an image as grayscale or RGB.
    :param filename: path of image file.
    :param representation: 1 for grayscale, 2 for RGB image.
    :return: image matrix.
    """
    image = imread(filename)
    flt_image = image / MAX_INTENSITY
    if representation == GRAY:  # gray
        return rgb2gray(flt_image)
    elif representation == RGB:  # RGB
        return flt_image

def show_image(image):
    """
    Presents an image.
    :param image: 3D matrix.
    """
    plt.imshow(image)
    plt.show()

def add_motion_blur(image, kernel_size, angle):
    """
    :param image: a RGB image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return:
    """
    # get kernel
    kernel = motion_blur_kernel(kernel_size, angle)
    kernel_3d = np.dstack((kernel, kernel, kernel))

    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = kernel.reshape((15, 15, 1, 1))

    # Convolve.
    image = tf.reshape(image, (3, image.shape[0], image.shape[1], 1))
    result = tf.nn.conv2d(image, gauss_kernel, padding="SAME", strides=[1,1,1,1])

    # for i in range(2):
    #     image[:,:,i] = tf.nn.conv2d(image[:,:,i], gauss_kernel, padding="SAME")

    # # convolve image.
    # for i in range(2):
    #     image[:,:,i] = convolve(image[:,:,i], kernel)
    return result

def add_motion_blur_image(image, kernel_size, angle):
    """
    :param image: a RGB image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return:
    """
    # get kernel
    kernel = motion_blur_kernel(kernel_size, angle)

    #kernel_3d = np.dstack((kernel, kernel, kernel))

    # Convolve.
    result = convolve(image,kernel)
    return result

img = "result.jpg"

if __name__ == '__main__':
    image = read_image(img,RGB).astype(np.float32)
    show_image(image)
    kernel = motion_blur_kernel(51,1)

    result_2 = add_motion_blur_1(image, 51, 1)


    # blured = add_motion_blur(image, 15, 1)
    # show_image(blured)

    kernel_3d = np.dstack((kernel, kernel, kernel)).astype(np.float32)

    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = tf.convert_to_tensor(kernel_3d)
    gauss_kernel = tf.reshape(gauss_kernel, (51, 51, 3, 1))
    # Convolve.
    # image = tf.reshape(image, [3, image.shape[1], image.shape[2], 1])

    print("image.shape before reshape", image.shape)


    image = tf.reshape(image, [1, image.shape[0], image.shape[1], 3])
    print("image.shape inside deblur", image.shape)
    print("gauss_kernel.shape inside deblur", gauss_kernel.shape)

    #result = tf.nn.conv2d(image, gauss_kernel, padding="SAME", strides=[1, 1, 1, 1])
    pointwise_filter = tf.eye(3, batch_shape=[1, 1])
    res = tf.squeeze(tf.nn.separable_conv2d(image, gauss_kernel, pointwise_filter, strides = [1, 1, 1, 1], padding="SAME"))
    # VALID means no padding
    with tf.Session() as sess:
        result = sess.run(res)

    print("result.shape", result.shape)



    #tf.reshape(result, (1024, 1024, 3))
    show_image(result)
    show_image(result_2)


