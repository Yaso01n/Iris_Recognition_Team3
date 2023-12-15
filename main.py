#starting imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
#machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# from skimage.feature import local_binary_pattern
# from skimage.feature import greycomatrix, greycoprops
# from skimage.measure import shannon_entropy

from Iris_localization import IrisLocalization
from Iris_normalization import IrisNormalization
from Apply_Filters import gabor_filter
from Apply_Filters import integro_diff_operator
from Code_Extraction import Iris_Code_Extraction


def hamming_distance_feature(iris_code_pair):
    distance_arr=[]
    for i in range(len(iris_code_pair)):
        for j in range(i + 1, len(iris_code_pair)):
            iris_code1= iris_code_pair[i]
            iris_code2 = iris_code_pair[j]

            # Check that the two iris codes have the same length
            if len(iris_code1) != len(iris_code2):
                raise ValueError("Iris codes must have the same length")

            # Calculate the Hamming distance
            distance = sum(bit1 != bit2 for bit1, bit2 in zip(iris_code1, iris_code2))
            distance_arr.append(distance)
    return distance_arr



# Define the database directory
training_data=[]
train_features=[]
# y_train = []

database_directory = "E:/4th year/MMU-Iris-Database"
IMG_SIZE = 240

# Iterate through folders in the database directory
for folder_number in os.listdir(database_directory):
    folder_path = os.path.join(database_directory, folder_number)
     # print(folder_path)
    # Iterate through files within each folder
    for folder_name in os.listdir(folder_path):
        folder_of_images = os.path.join(folder_path, folder_name)
        # print(folder_of_images)
        # Iterate through files within each folder
        for img in os.listdir(folder_of_images):
            try:
                img_array = cv2.imread(os.path.join(folder_of_images,img) ,cv2.IMREAD_GRAYSCALE) 
                iris, coordinates = IrisLocalization(img_array)
                res_image = IrisNormalization(img_array, coordinates, iris)
                iris_code=Iris_Code_Extraction(res_image)
                train_features.append(iris_code)
                print('good')
            except Exception as e:
                print("problem")
                break    
print(len(training_data))
print(img_array.shape[1])

#separating image's array and labels of them
X = []
y=[]
dist_train=hamming_distance_feature(train_features)
X=np.array(dist_train)
X = X.reshape(X.shape[0], -1)
# # X = X.reshape(X.shape[0], -1)
         
print(X)
for i in range(len(dist_train)):
    if (dist_train[i]<30000):
        y.append(1)    #Matching
    elif (dist_train[i]>=30000):
        y.append(0)    #Non-Matching    
print(y)
    


# # Define the database directory
# testing_data=[]
# test_features=[]
# y_test = []

# database_directory2 = "E:/4th year/Biometrics/data/test"
# IMG_SIZE = 240

# # Iterate through folders in the database directory
# for folder_number2 in os.listdir(database_directory2):
#     folder_path2 = os.path.join(database_directory2, folder_number2)
#     for img in os.listdir(folder_path2):
#         try:
#             img_array = cv2.imread(os.path.join(folder_path2,img) ,cv2.IMREAD_GRAYSCALE) 
#             iris, coordinates = IrisLocalization(img_array)
#             res_image = IrisNormalization(img_array, coordinates, iris)
#             iris_code=Iris_Code_Extraction(res_image)
#             test_features.append(iris_code)
#             print('good')
#         except Exception as e:
#             print("problem")
#             break    
# print(len(training_data))
# print(img_array.shape[1])

# #separating image's array and labels of them
# X_test = []

# dist_test=hamming_distance_feature(test_features)
# X_test=np.array(dist_test)
# X_test = X_test.reshape(X_test.shape[0], -1)
# for i in range(len(dist_test)):
#     if (dist_test[i]<2000):
#         y_test.append(1)    #Matching
#     elif (dist_test[i]>=2000):
#         y_test.append(0)    #Non-Matching 


# Split the data into training and testing sets
# You may also consider cross-validation for model evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a machine learning model (e.g., Support Vector Machine)
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)




































































































