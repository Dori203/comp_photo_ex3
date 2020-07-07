import argparse
import pickle
import os
import imageio
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import cv2

from stylegan import dnnlib as dnnlib
import stylegan.dnnlib.tflib as tflib
import stylegan.config as config

from perceptual_model import PerceptualModel
from scipy.ndimage.filters import convolve
from skimage.draw import line
import random
import math as m



# STYLEGAN_MODEL_URL = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
STYLEGAN_MODEL_URL = 'https://drive.google.com/uc?export=download&id=1vUpawbqkcaS2jM_Q0DLfL83dk1mLx_wl'
KERNEL_SIZE = 15
KERNEL_SIZES = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51]

def generate_random_mask(img_shape, mask_size):
    mask_2d = np.ones(img_shape, dtype=np.uint8)

    vq = img_shape[0] // 4
    top = np.random.randint(low=vq, high=3 * vq - mask_size[0])

    hq = img_shape[1] // 4
    left = np.random.randint(low=hq, high=3 * hq - mask_size[1])

    mask_2d[top:top + mask_size[0], left:left + mask_size[1]] = 0

    return mask_2d[..., np.newaxis]

def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.
    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    pi = tf.constant(m.pi)
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size-1)
    else:
        alpha = np.tan(pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2*half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size-1 - p1[0], kernel_size-1 - p1[1])
        else:
            alpha = np.tan(pi * 0.5 * (1-norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2*half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size-1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel

def random_motion_blur(list_of_kernel_sizes):
    """
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return:
    """
    # sample an angle and kernel size uniformaly.
    alpha = random.uniform(0, np.pi)
    index = random.randrange(0, len(list_of_kernel_sizes))
    # return chosen values to image.
    return np.array([list_of_kernel_sizes[index], alpha])


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
    gauss_kernel = tf.convert_to_tensor(kernel_3d)
    gauss_kernel = tf.reshape(gauss_kernel,(kernel_size, kernel_size, 3, 1))
    # Convolve.
    #image = tf.reshape(image, [-1, image.shape[1], image.shape[2], 3])
    #image = tf.expand_dims(image, 0)


    pointwise_filter = tf.eye(3, batch_shape=[1, 1])
    result = tf.nn.separable_conv2d(image, gauss_kernel, pointwise_filter, padding="SAME", strides=[1,1,1,1])
    return result

def add_motion_blur_single_image(image, kernel_size, angle):
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


def optimize_latent_codes(args):
    tflib.init_tf()

    with dnnlib.util.open_url(STYLEGAN_MODEL_URL, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    latent_code = tf.get_variable(
        name='latent_code', shape=(1, 18, 512), dtype='float32', initializer=tf.initializers.zeros()
    )

    generated_img = Gs.components.synthesis.get_output_for(latent_code, randomize_noise=True)
    generated_img = tf.transpose(generated_img, [0, 2, 3, 1])
    generated_img = ((generated_img + 1) / 2) * 255

    original_img = tf.placeholder(tf.float32, [None, args.input_img_size[0], args.input_img_size[1], 3])
    blur_parameters = tf.placeholder(tf.float32, [None, 2])

    degraded_img_resized_for_perceptual = tf.image.resize_images(
        add_motion_blur(original_img,blur_parameters[0],blur_parameters[1]), tuple(args.perceptual_img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    print("degraded_img_resized_for_perceptual shape is: ", degraded_img_resized_for_perceptual.shape)

    generated_img_resized_to_original = tf.image.resize_images(
        generated_img, tuple(args.input_img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    print("generated_img_resized_to_original shape is: ", generated_img_resized_to_original.shape)

    generated_img_resized_for_perceptual = tf.image.resize_images(
        add_motion_blur(generated_img_resized_to_original,blur_parameters[0],blur_parameters[1]), tuple(args.perceptual_img_size),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    print("generated_img_resized_for_perceptual shape is: ", generated_img_resized_for_perceptual.shape)


    generated_img_for_display = tf.saturate_cast(generated_img_resized_to_original, tf.uint8)

    print("generated image shape is: ", generated_img_for_display.shape)

    perceptual_model = PerceptualModel(img_size=args.perceptual_img_size)
    generated_img_features = perceptual_model(generated_img_resized_for_perceptual)

    target_img_features = perceptual_model(degraded_img_resized_for_perceptual)

    loss_op = tf.reduce_mean(tf.abs(generated_img_features - target_img_features))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=[latent_code])

    sess = tf.get_default_session()

    img_names = sorted(os.listdir(args.imgs_dir))
    for img_name in img_names:
        img = imageio.imread(os.path.join(args.imgs_dir, img_name))
        img = cv2.resize(img, dsize=tuple(args.input_img_size))
        blur_parameters = random_motion_blur(KERNEL_SIZES)
        corrupted_img = add_motion_blur_single_image(img,blur_parameters[0],blur_parameters[1])
        print("blur parameters are: ", blur_parameters[0], " ", blur_parameters[1])

        imageio.imwrite(os.path.join(args.corruptions_dir, img_name), corrupted_img)
        #imageio.imwrite(os.path.join(args.masks_dir, img_name), mask * 255)

        sess.run(tf.variables_initializer([latent_code] + optimizer.variables()))

        progress_bar_iterator = tqdm(
            iterable=range(args.total_iterations),
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}',
            desc=img_name
        )

        for i in progress_bar_iterator:
            loss, _ = sess.run(
                fetches=[loss_op, train_op],
                feed_dict={
                    original_img: img[np.newaxis, ...],
                    blur_parameters: blur_parameters[np.newaxis, ...]
                }
            )

            progress_bar_iterator.set_postfix_str('loss=%.2f' % loss)

        reconstructed_imgs, latent_codes = sess.run(
            fetches=[generated_img_for_display, latent_code],
            feed_dict={
                original_img: img[np.newaxis, ...],
                blur_parameters: blur_parameters[np.newaxis, ...]
            }
        )

        imageio.imwrite(os.path.join(args.restorations_dir, img_name), reconstructed_imgs[0])
        np.savez(file=os.path.join(args.latents_dir, img_name + '.npz'), latent_code=latent_codes[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs-dir', type=str, required=True)
    parser.add_argument('--masks-dir', type=str, required=True)
    parser.add_argument('--corruptions-dir', type=str, required=True)
    parser.add_argument('--restorations-dir', type=str, required=True)
    parser.add_argument('--latents-dir', type=str, required=True)

    parser.add_argument('--input-img-size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--perceptual-img-size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--mask-size', type=int, nargs=2, default=(5, 5))
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--total-iterations', type=int, default=1000)

    args = parser.parse_args()

    os.makedirs(args.masks_dir, exist_ok=True)
    os.makedirs(args.corruptions_dir, exist_ok=True)
    os.makedirs(args.restorations_dir, exist_ok=True)
    os.makedirs(args.latents_dir, exist_ok=True)

    optimize_latent_codes(args)