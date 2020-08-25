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



STYLEGAN_MODEL_URL = 'https://drive.google.com/uc?export=download&id=1vUpawbqkcaS2jM_Q0DLfL83dk1mLx_wl'

def motion_blur_kernel_1D(kernel_size, angle):
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
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel

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
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    kernel_3d = np.dstack((kernel, kernel, kernel))
    return kernel_3d

def add_motion_blur(image, kernel_3d):
    """
    :param image: a RGB image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return:
    """
    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    #gauss_kernel = tf.convert_to_tensor(kernel_3d)
    kernel_3d = tf.reshape(kernel_3d,(kernel_3d.shape[1], kernel_3d.shape[2], kernel_3d.shape[3], 1))
    # Convolve.
    pointwise_filter = tf.eye(3, batch_shape=[1, 1])
    result = tf.nn.separable_conv2d(image, kernel_3d, pointwise_filter, padding="SAME", strides=[1,1,1,1])
    return result

def add_motion_blur_single_image(image, kernel_size, angle):
    """
    :param image: a RGB image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return:
    """
    # get kernel
    kernel = motion_blur_kernel_1D(kernel_size, angle)

    # convolve image.
    for i in range(3):
        image[:,:,i] = convolve(image[:,:,i], kernel, mode='nearest')
    return image

def get_image_from_latant_code(latent_code):
    tflib.init_tf()
    with dnnlib.util.open_url(STYLEGAN_MODEL_URL, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    print(latent_code.shape)

    generated_img = Gs.components.synthesis.get_output_for(latent_code, randomize_noise=False)
    generated_img = tf.transpose(generated_img, [0, 2, 3, 1])
    generated_img = ((generated_img + 1) / 2) * 255
    return generated_img

def optimize_latent_codes(args):
    tflib.init_tf()

    with dnnlib.util.open_url(STYLEGAN_MODEL_URL, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    latent_code = tf.get_variable(
        name='latent_code', shape=(1, 18, 512), dtype='float32', initializer=tf.initializers.zeros()
    )

    generated_img = Gs.components.synthesis.get_output_for(latent_code, randomize_noise=False)
    generated_img = tf.transpose(generated_img, [0, 2, 3, 1])
    generated_img = ((generated_img + 1) / 2) * 255

    original_img = tf.placeholder(tf.float32, [None, args.input_img_size[0], args.input_img_size[1], 3])
    blur_kernel = tf.placeholder(tf.float32, [None, args.blur_parameters[0], args.blur_parameters[0], 3])

    degraded_img_resized_for_perceptual = tf.image.resize_images(
        add_motion_blur(original_img,blur_kernel), tuple(args.perceptual_img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )


    generated_img_resized_to_original = tf.image.resize_images(
        generated_img, tuple(args.input_img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )


    generated_img_resized_for_perceptual = tf.image.resize_images(
        add_motion_blur(generated_img_resized_to_original, blur_kernel), tuple(args.perceptual_img_size),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)



    generated_img_for_display = tf.saturate_cast(generated_img_resized_to_original, tf.uint8)


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
        blur_kernel_3d = motion_blur_kernel(args.blur_parameters[0], args.blur_parameters[1])
        corrupted_img = add_motion_blur_single_image(img,args.blur_parameters[0],args.blur_parameters[1])

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
                    blur_kernel: blur_kernel_3d[np.newaxis, ...]
                }
            )

            progress_bar_iterator.set_postfix_str('loss=%.2f' % loss)

        reconstructed_imgs, latent_codes = sess.run(
            fetches=[generated_img_for_display, latent_code],
            feed_dict={
                original_img: img[np.newaxis, ...],
                blur_kernel: blur_kernel_3d[np.newaxis, ...]
            }
        )

        imageio.imwrite(os.path.join(args.restorations_dir, img_name), reconstructed_imgs[0])
        np.savez(file=os.path.join(args.latents_dir, img_name + '.npz'), latent_code=latent_codes[0])

        latent_code = latent_codes[0].reshape((1, 18, 512))
        print("latent code shape is: ", latent_code)
        print("latent code value is: ", latent_code)
        latent_1 = get_image_from_latant_code(latent_code)
        # latent_2 = get_image_from_latant_code(latent_codes[1])
        imageio.imwrite(os.path.join(args.restorations_dir, "latent_0.png"), latent_1)
        # imageio.imwrite(os.path.join(args.restorations_dir, "latent_1.png"), latent_2)


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
    parser.add_argument('--blur-parameters', type=int, nargs=2, default=(51, 1))
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--total-iterations', type=int, default=100)

    args = parser.parse_args()

    os.makedirs(args.masks_dir, exist_ok=True)
    os.makedirs(args.corruptions_dir, exist_ok=True)
    os.makedirs(args.restorations_dir, exist_ok=True)
    os.makedirs(args.latents_dir, exist_ok=True)

    optimize_latent_codes(args)