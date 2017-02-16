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

def plot_learning_curve(title, train_sizes, train_scores, test_scores, ylim=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()