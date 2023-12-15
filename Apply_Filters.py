import numpy as np
from scipy import ndimage


def integro_diff_operator(image, sigma):
    # Apply Gaussian smoothing to the image
    smoothed = ndimage.gaussian_filter(image, sigma)

    # Compute gradients using Sobel operators
    gradient_x = ndimage.sobel(smoothed, axis=1)
    gradient_y = ndimage.sobel(smoothed, axis=0)

    # Compute the magnitude of the gradients
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute the orientation of the gradients
    orientation = np.arctan2(gradient_y, gradient_x)

    return magnitude, orientation


def gabor_filter(image, sigma, theta, frequency):
    # Apply Gaussian smoothing to the image
    smoothed = ndimage.gaussian_filter(image, sigma)

    # Construct the Gabor kernel
    kernel_size = int(4 * sigma)
    kernel = np.zeros((kernel_size, kernel_size))
    for x in range(kernel_size):
        for y in range(kernel_size):
            x_prime = x * np.cos(theta) + y * np.sin(theta)
            y_prime = -x * np.sin(theta) + y * np.cos(theta)
            kernel[x, y] = np.exp(-0.5 * (x_prime**2 + frequency**2 * y_prime**2) / sigma**2) * np.cos(2 * np.pi * frequency * x_prime)

    # Apply the Gabor kernel to the image
    filtered = ndimage.convolve(smoothed, kernel)

    return filtered