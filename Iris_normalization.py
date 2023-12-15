import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import math

def IrisNormalization(image, inner_circle, outer_circle):
  """
  Normalizes an iris image.

  Args:
    image: A grayscale image of the iris.
    inner_circle: A tuple containing the (x, y, radius) of the inner circle of the iris.
    outer_circle: A tuple containing the (x, y, radius) of the outer circle of the iris.

  Returns:
    A normalized iris image.
  """

  # Localized the iris image.
  localized_img = image

  # Define the dimensions of the normalized iris image.
  row = 64
  col = 512

  # Create an empty array to store the normalized iris image.
  normalized_iris = np.zeros(shape=(row, col))

  # Get the x and y coordinates of the inner and outer circle boundaries.
  inner_x, inner_y, inner_radius = inner_circle
  outer_x, outer_y, outer_radius = outer_circle

  # Compute the angle between each pixel on the inner and outer circle boundaries.
  angle = 2.0 * math.pi / col

  # Create arrays to store the x and y coordinates of the inner and outer circle boundaries.
  inner_boundary_x = np.zeros(shape=(1, col))
  inner_boundary_y = np.zeros(shape=(1, col))
  outer_boundary_x = np.zeros(shape=(1, col))
  outer_boundary_y = np.zeros(shape=(1, col))

  # Iterate over the pixels on the inner and outer circle boundaries and compute their x and y coordinates.
  for j in range(col):
    inner_boundary_x[0][j] = inner_x + inner_radius * math.cos(angle * j)
    inner_boundary_y[0][j] = inner_y + inner_radius * math.sin(angle * j)

    outer_boundary_x[0][j] = outer_x + outer_radius * math.cos(angle * j)
    outer_boundary_y[0][j] = outer_y + outer_radius * math.sin(angle * j)

  # Iterate over the pixels in the normalized iris image.
  for j in range(512):
    for i in range(64):
      # Compute the x and y coordinates of the corresponding pixel in the localized iris image.
      localized_x = min(int(int(inner_boundary_x[0][j])
                              + (int(outer_boundary_x[0][j]) - int(inner_boundary_x[0][j])) * (i / 64.0)), localized_img.shape[1] - 1)
      localized_y = min(int(int(inner_boundary_y[0][j])
                              + (int(outer_boundary_y[0][j]) - int(inner_boundary_y[0][j])) * (i / 64.0)), localized_img.shape[0] - 1)

      # Get the pixel value at the corresponding coordinates in the localized iris image.
      localized_pixel_value = localized_img[localized_y][localized_x]

      # Set the pixel value at the corresponding coordinates in the normalized iris image.
      normalized_iris[i][j] = localized_pixel_value

  # Invert the normalized iris image.
  res_image = 255 - normalized_iris

  # Return the normalized iris image.
  return res_image