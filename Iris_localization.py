import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from skimage.transform import hough_circle, hough_circle_peaks
import math
from PIL import Image


def IrisLocalization(eye):
    """
    Localizes the iris and pupil in an eye image.

    Args:
    eye: A grayscale image of the eye.

    Returns:
    A tuple containing two NumPy arrays:
        - The first array is the iris center and radius, in the form (x, y, r).
        - The second array is the pupil center and radius, in the form (x, y, r).
    """

    # Blur the image.
    blured = cv2.bilateralFilter(eye, 9, 100, 100)

    # Find the minimum value of the pixels in the blurred image, along the x and y axes.
    Xp = blured.sum(axis=0).argmin()
    Yp = blured.sum(axis=1).argmin()

    # Find the index of the minimum value of the pixels in a square of pixels centered at (Xp, Yp).
    x = blured[max(Yp - 60, 0):min(Yp + 60, 300), max(Xp - 60, 0):min(Xp + 60, 400)].sum(axis=0).argmin()
    y = blured[max(Yp - 60, 0):min(Yp + 60, 300), max(Xp - 60, 0):min(Xp + 60, 400)].sum(axis=1).argmin()

    # Adjust the x and y coordinates of the pupil center.
    Xp = max(Xp - 60, 0) + x
    Yp = max(Yp - 60, 0) + y

    # Set a default value for rp
    rp = 0
    xp = 0 
    yp = 0

    # Use the Hough transform to find the pupil.
    if Xp >= 20 and Yp >= 10:
        blur = cv2.GaussianBlur(eye[Yp - 60:Yp + 60, Xp - 60:Xp + 60], (5, 5), 0)

        # Preprocess the region of interest
        equ = cv2.equalizeHist(blur)
        _, thresh = cv2.threshold(equ, 100, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV)

        pupil_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12, minRadius=15, maxRadius=80)
        # pupil_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12, minRadius=15, maxRadius=80)

        # Check if circles were detected
        if pupil_circles is not None and len(pupil_circles[0]) > 0:
            # Get the x, y, and radius of the pupil.
            xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
            print(rp)
            # Adjust the x and y coordinates of the pupil center.
            xp = Xp - 60 + xp
            yp = Yp - 60 + yp
        else:
            # Handle the case when no circles are detected
            print("No pupil circles detected in the region.")
            # You might want to return some default values or skip further processing
    else:
        pupil_circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 4, 280, minRadius=25, maxRadius=55, param2=51)

        # Check if circles were detected
        if pupil_circles is not None and len(pupil_circles[0]) > 0:
            # Get the x, y, and radius of the pupil.
            xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
        else:
            # Handle the case when no circles are detected
            print("No pupil circles detected in the image.")
            # You might want to return some default values or skip further processing

    # Make a copy of the eye image.
    eye_copy = eye.copy()

    # Slightly enlarge the pupil radius to improve the results.
    rp = rp + 7
    print(rp)
    print(xp)
    print(yp)
    # Blur the eye image.
    blured_copy = cv2.medianBlur(eye_copy, 11)
    blured_copy = cv2.medianBlur(blured_copy, 11)
    blured_copy = cv2.medianBlur(blured_copy, 11)

    # Compute the Canny edges of the blurred eye image.
    eye_edges = cv2.Canny(blured_copy, threshold1=15, threshold2=30, L2gradient=True)

    # Set the Canny edges to zero in a region around the pupil.
    eye_edges[:, xp - rp - 30:xp + rp + 30] = 0

    # Hough transform to find the iris.
    hough_radii = np.arange(rp + 30, 100, 2)
    hough_res = hough_circle(eye_edges, hough_radii)
    accums, xi, yi, ri = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    # Iris center
    iris = []
    iris.extend(xi)
    iris.extend(yi)
    iris.extend(ri)
    if ((iris[0] - xp) ** 2 + (iris[1] - yp) ** 2) ** 0.5 > rp * 0.3:
        iris[0] = xp
        iris[1] = yp

    return np.array(iris), np.array([xp, yp, rp])






# img = cv2.imread('E:/4th year/Biometrics/Images/aisha_waziery/IMG_20231023_131928.jpg')
img = cv2.imread('E:/4th year/Biometrics/CLASSES_400_300_Part1/C4_S1_I5.tiff') 
# img = cv2.imread('C:/Users/pc/Desktop/L3.jpg')

img_grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_grey.shape)
# Specify the desired size (width, height)
plt.imshow(img_grey, cmap='gray')
plt.show()
# Crop the image to the desired size
# cropped_image = image.crop((0, 0, desired_size[0], desired_size[1]))
# cropped_image=img_grey[1200:2500, 1000:4000]
# print(cropped_image.shape)
# plt.imshow(cropped_image, cmap='gray')
# plt.show()
# # Display or save the cropped image as needed
# cv2.imshow('Cropped Image', cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





iris, coordinates = IrisLocalization(img_grey)

cv2.circle(img_grey,(iris[0],iris[1]),iris[2],(255,255,255),2)
cv2.circle(img_grey,(iris[0],iris[1]),2,(255,255,255),3)

cv2.circle(img_grey,(coordinates[0],coordinates[1]),coordinates[2],(255,255,255),2)
cv2.circle(img_grey,(coordinates[0],coordinates[1]),2,(255,255,255),3)

plt.imshow(img_grey, cmap='gray')
plt.show()

# print(iris)
# print(iris.ndim)

# print(coordinates)
# print(coordinates.ndim)





# def IrisLocalization(eye):
#     """
#     Localizes the iris and pupil in an eye image.

#     Args:
#     eye: A grayscale image of the eye.

#     Returns:
#     A tuple containing two NumPy arrays:
#         - The first array is the iris center and radius, in the form (x, y, r).
#         - The second array is the pupil center and radius, in the form (x, y, r).
#     """

#     # Blur the image.
#     blured = cv2.bilateralFilter(eye, 9, 100, 100)

#     # Find the minimum value of the pixels in the blurred image, along the x and y axes.
#     Xp = blured.sum(axis=0).argmin()
#     Yp = blured.sum(axis=1).argmin()

#     # Find the index of the minimum value of the pixels in a square of pixels centered at (Xp, Yp).
#     x = blured[max(Yp - 60, 0):min(Yp + 60, 1064), max(Xp - 60, 0):min(Xp + 60, 1909)].sum(axis=0).argmin()
#     y = blured[max(Yp - 60, 0):min(Yp + 60, 1064), max(Xp - 60, 0):min(Xp + 60, 1909)].sum(axis=1).argmin()

#     # Adjust the x and y coordinates of the pupil center.
#     Xp = max(Xp - 60, 0) + x
#     Yp = max(Yp - 60, 0) + y

#     # Set a default value for rp
#     rp = 0
#     xp = 0 
#     yp = 0

#     # Use the Hough transform to find the pupil.
#     if Xp >= 300 and Yp >= 150:
#         blur = cv2.GaussianBlur(eye[Yp - 60:Yp + 60, Xp - 60:Xp + 60], (5, 5), 0)

#         # Preprocess the region of interest
#         equ = cv2.equalizeHist(blur)
#         _, thresh = cv2.threshold(equ, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV)

#         pupil_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12, minRadius=15, maxRadius=80)
#         # pupil_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12, minRadius=15, maxRadius=80)

#         # Check if circles were detected
#         if pupil_circles is not None and len(pupil_circles[0]) > 0:
#             # Get the x, y, and radius of the pupil.
#             xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
#             print(rp)
#             # Adjust the x and y coordinates of the pupil center.
#             xp = Xp  + xp
#             yp = Yp + 10 + yp
#         else:
#             # Handle the case when no circles are detected
#             print("No pupil circles detected in the region.")
#             # You might want to return some default values or skip further processing
#     else:
#         pupil_circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 4, 280, minRadius=25, maxRadius=55, param2=51)

#         # Check if circles were detected
#         if pupil_circles is not None and len(pupil_circles[0]) > 0:
#             # Get the x, y, and radius of the pupil.
#             xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
#         else:
#             # Handle the case when no circles are detected
#             print("No pupil circles detected in the image.")
#             # You might want to return some default values or skip further processing

#     # Make a copy of the eye image.
#     eye_copy = eye.copy()

#     # Slightly enlarge the pupil radius to improve the results.
#     rp = rp + 55
#     print(rp)
#     print(xp)
#     print(yp)
#     # Blur the eye image.
#     blured_copy = cv2.medianBlur(eye_copy, 11)
#     blured_copy = cv2.medianBlur(blured_copy, 11)
#     blured_copy = cv2.medianBlur(blured_copy, 11)

#     # Compute the Canny edges of the blurred eye image.
#     eye_edges = cv2.Canny(blured_copy, threshold1=15, threshold2=30, L2gradient=True)

#     # Set the Canny edges to zero in a region around the pupil.
#     eye_edges[:, xp - rp - 150:xp + rp + 150] = 0

#     # Hough transform to find the iris.
#     hough_radii = np.arange(rp + 125, 275, 2)
#     hough_res = hough_circle(eye_edges, hough_radii)
#     accums, xi, yi, ri = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

#     # Iris center
#     iris = []
#     iris.extend(xi)
#     iris.extend(yi)
#     iris.extend(ri)
#     if ((iris[0] - xp) ** 2 + (iris[1] - yp) ** 2) ** 0.5 > rp * 0.3:
#         iris[0] = xp
#         iris[1] = yp

#     return np.array(iris), np.array([xp, yp, rp])






#=================================================================
# def IrisLocalization(eye):
#     """
#     Localizes the iris and pupil in an eye image.

#     Args:
#     eye: A grayscale image of the eye.

#     Returns:
#     A tuple containing two NumPy arrays:
#         - The first array is the iris center and radius, in the form (x, y, r).
#         - The second array is the pupil center and radius, in the form (x, y, r).
#     """

#     # Blur the image.
#     blured = cv2.bilateralFilter(eye, 9, 100, 100)

#     # Find the minimum value of the pixels in the blurred image, along the x and y axes.
#     Xp = blured.sum(axis=0).argmin()
#     Yp = blured.sum(axis=1).argmin()

#     # Find the index of the minimum value of the pixels in a square of pixels centered at (Xp, Yp).
#     x = blured[max(Yp - 60, 0):min(Yp + 60, 300), max(Xp - 60, 0):min(Xp + 60, 400)].sum(axis=0).argmin()
#     y = blured[max(Yp - 60, 0):min(Yp + 60, 300), max(Xp - 60, 0):min(Xp + 60, 400)].sum(axis=1).argmin()

#     # Adjust the x and y coordinates of the pupil center.
#     Xp = max(Xp - 60, 0) + x
#     Yp = max(Yp - 60, 0) + y

#     # Set a default value for rp
#     rp = 0
#     xp = 0 
#     yp = 0

#     # Use the Hough transform to find the pupil.
#     if Xp >= 20 and Yp >= 10:
#         blur = cv2.GaussianBlur(eye[Yp - 60:Yp + 60, Xp - 60:Xp + 60], (5, 5), 0)

#         # Preprocess the region of interest
#         equ = cv2.equalizeHist(blur)
#         _, thresh = cv2.threshold(equ, 100, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV)

#         pupil_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12, minRadius=15, maxRadius=80)
#         # pupil_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12, minRadius=15, maxRadius=80)

#         # Check if circles were detected
#         if pupil_circles is not None and len(pupil_circles[0]) > 0:
#             # Get the x, y, and radius of the pupil.
#             xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
#             print(rp)
#             # Adjust the x and y coordinates of the pupil center.
#             xp = Xp - 60 + xp
#             yp = Yp - 60 + yp
#         else:
#             # Handle the case when no circles are detected
#             print("No pupil circles detected in the region.")
#             # You might want to return some default values or skip further processing
#     else:
#         pupil_circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 4, 280, minRadius=25, maxRadius=55, param2=51)

#         # Check if circles were detected
#         if pupil_circles is not None and len(pupil_circles[0]) > 0:
#             # Get the x, y, and radius of the pupil.
#             xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
#         else:
#             # Handle the case when no circles are detected
#             print("No pupil circles detected in the image.")
#             # You might want to return some default values or skip further processing

#     # Make a copy of the eye image.
#     eye_copy = eye.copy()

#     # Slightly enlarge the pupil radius to improve the results.
#     rp = rp + 7
#     print(rp)
#     print(xp)
#     print(yp)
#     # Blur the eye image.
#     blured_copy = cv2.medianBlur(eye_copy, 11)
#     blured_copy = cv2.medianBlur(blured_copy, 11)
#     blured_copy = cv2.medianBlur(blured_copy, 11)

#     # Compute the Canny edges of the blurred eye image.
#     eye_edges = cv2.Canny(blured_copy, threshold1=15, threshold2=30, L2gradient=True)

#     # Set the Canny edges to zero in a region around the pupil.
#     eye_edges[:, xp - rp - 30:xp + rp + 30] = 0

#     # Hough transform to find the iris.
#     hough_radii = np.arange(rp + 30, 100, 2)
#     hough_res = hough_circle(eye_edges, hough_radii)
#     accums, xi, yi, ri = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

#     # Iris center
#     iris = []
#     iris.extend(xi)
#     iris.extend(yi)
#     iris.extend(ri)
#     if ((iris[0] - xp) ** 2 + (iris[1] - yp) ** 2) ** 0.5 > rp * 0.3:
#         iris[0] = xp
#         iris[1] = yp

#     return np.array(iris), np.array([xp, yp, rp])




#================================================================================================================================================
# def IrisLocalization(eye):
#     """
#     Localizes the iris and pupil in an eye image.

#     Args:
#     eye: A grayscale image of the eye.

#     Returns:
#     A tuple containing two NumPy arrays:
#         - The first array is the iris center and radius, in the form (x, y, r).
#         - The second array is the pupil center and radius, in the form (x, y, r).
#     """

#     # Blur the image.
#     blured = cv2.bilateralFilter(eye, 9, 100, 100)

#     # Find the minimum value of the pixels in the blurred image, along the x and y axes.
#     Xp = blured.sum(axis=0).argmin()
#     Yp = blured.sum(axis=1).argmin()
#     plt.imshow(blured, cmap='gray')
#     plt.show()
#     # Find the index of the minimum value of the pixels in a square of pixels centered at (Xp, Yp).
#     x = blured[max(Yp - 60, 0):min(Yp + 60, 1300), max(Xp - 60, 0):min(Xp + 60, 3000)].sum(axis=0).argmin()
#     y = blured[max(Yp - 60, 0):min(Yp + 60, 1300), max(Xp - 60, 0):min(Xp + 60, 3000)].sum(axis=1).argmin()

#     # Adjust the x and y coordinates of the pupil center.
#     Xp = max(Xp - 60, 0) + x
#     Yp = max(Yp - 60, 0) + y
#     print(Xp)
#     print(Yp)
#     # Set a default value for rp
#     rp = 0
#     xp = 0 
#     yp = 0

#     # Use the Hough transform to find the pupil.
#     if Xp >= 300 and Yp >= 150:
#         blur = cv2.GaussianBlur(eye[Yp - 60:Yp + 60, Xp - 60:Xp + 60], (5, 5), 0)
#         print('1')
#         # Preprocess the region of interest
#         equ = cv2.equalizeHist(blur)
#         _, thresh = cv2.threshold(equ, 80, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV)
#         plt.imshow(thresh, cmap='gray')
#         plt.show()
#         pupil_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12, minRadius=15, maxRadius=80)
#         # pupil_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12, minRadius=15, maxRadius=80)

#         # Check if circles were detected
#         if pupil_circles is not None and len(pupil_circles[0]) > 0:
#             # Get the x, y, and radius of the pupil.
#             xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
#             print(rp)
#             # Adjust the x and y coordinates of the pupil center.
#             xp = Xp -875+ xp
#             yp = Yp + yp
#         else:
#             # Handle the case when no circles are detected
#             print("No pupil circles detected in the region.")
#             # You might want to return some default values or skip further processing
#     else:
#         pupil_circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 4, 280, minRadius=25, maxRadius=55, param2=51)

#         # Check if circles were detected
#         if pupil_circles is not None and len(pupil_circles[0]) > 0:
#             # Get the x, y, and radius of the pupil.
#             xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
#         else:
#             # Handle the case when no circles are detected
#             print("No pupil circles detected in the image.")
#             # You might want to return some default values or skip further processing

#     # Make a copy of the eye image.
#     eye_copy = eye.copy()

#     # Slightly enlarge the pupil radius to improve the results.
#     rp = rp + 70
#     print(rp)
#     print(xp)
#     print(yp)
#     # Blur the eye image.
#     blured_copy = cv2.medianBlur(eye_copy, 11)
#     blured_copy = cv2.medianBlur(blured_copy, 11)
#     blured_copy = cv2.medianBlur(blured_copy, 11)

#     # Compute the Canny edges of the blurred eye image.
#     eye_edges = cv2.Canny(blured_copy, threshold1=15, threshold2=30, L2gradient=True)

#     # Set the Canny edges to zero in a region around the pupil.
#     eye_edges[:, xp - rp - 300:xp + rp + 300] = 0

#     # Hough transform to find the iris.
#     hough_radii = np.arange(rp + 200, 500, 20)
#     hough_res = hough_circle(eye_edges, hough_radii)
#     accums, xi, yi, ri = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

#     # Iris center
#     iris = []
#     iris.extend(xi)
#     iris.extend(yi)
#     iris.extend(ri)
#     if ((iris[0] - xp) ** 2 + (iris[1] - yp) ** 2) ** 0.5 > rp * 0.3:
#         iris[0] = xp
#         iris[1] = yp

#     return np.array(iris), np.array([xp, yp, rp])