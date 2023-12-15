import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from skimage.transform import hough_circle, hough_circle_peaks
from Apply_Filters import gabor_filter
from Apply_Filters import integro_diff_operator

def Iris_Code_Extraction(res_image):
    # Apply Daugman's integro-differential operator
    magnitude, orientation = integro_diff_operator(res_image, 3)

    # Apply Gabor filtering on the normalized image
    gabor_filtered = gabor_filter(res_image, 3, np.pi/4, 0.5)

    # Extract features from the filtered image
    feature_vector = np.concatenate((gabor_filtered.flatten(), orientation.flatten(), magnitude.flatten()))
    template=feature_vector
    threshold = 15 # Adjust the threshold as needed
    iris_code = encode_feature_vector(template, threshold)
    return iris_code


def encode_feature_vector(feature_vector, threshold):
    # Apply thresholding to the feature vector
    binary_features = np.where(feature_vector >= threshold, 1, 0)

    # Convert the binary features into a binary iris code
    binary_iris_code = "".join(map(str, binary_features))

    return binary_iris_code
