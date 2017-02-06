import scipy.misc
import numpy as np

def noise(noise_level, shape):
    return np.random.choice([0, 1], shape, p=[1-noise_level, noise_level])

def noise_masks(noise_levels, shape):
    return [noise(noise_level, shape) for noise_level in noise_levels]

def noisy_images(image_filename, min_noise, max_noise, count):
    image = scipy.misc.imread(image_filename, flatten=True)/255
    noise_levels = np.linspace(min_noise, max_noise, count)
    return [np.logical_xor(image, noise_mask) for noise_mask in noise_masks(noise_levels, np.shape(image))]

def random_images(min_noise, max_noise, count, shape):
    noise_levels = np.linspace(min_noise, max_noise, count)
    return noise_masks(noise_levels, shape)

def flatten(matrix_array):
    return [matrix.flatten() for matrix in matrix_array]

def generate_dataset():
    import os    
    files = os.listdir('patterns')

    count = 100

    images = []
    for file in files:
        images += noisy_images('patterns/' + file, 0, 0.4, count)
    images += random_images(0.1, 0.9, count, np.shape(images[0]))

    labels = []
    for i in range(len(files) + 1):
        labels += [i] * count

    return flatten(images), labels