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
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
# from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sklearn.metrics.pairwise import cosine_similarity




def extract_eye_features(gray_image):
    # Local Binary Pattern (LBP) feature extraction
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)  # Normalize the histogram
    # Shannon Entropy feature extraction
    entropy = shannon_entropy(gray_image)
    # Combine all features into a single feature vector
    features = np.concatenate([lbp_hist, [entropy]])
    return features



def readimages(database_directory):
    list_fetures_data=[]
    # IMG_SIZE = 240                
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
    #             file_path = os.path.join(folder_path, file_name)

                            
               
                try:
                    img_array = cv2.imread(os.path.join(folder_of_images,img) ,cv2.IMREAD_GRAYSCALE)  
                    features = extract_eye_features(img_array)
                    # Reshape the feature vectors to be 2D arrays (required for cosine_similarity)
                    features1 = features.reshape(1, -1)
                    # Calculate cosine similarity between the feature vectors
                    similarity_score = cosine_similarity(features1, features1)[0, 0]

                    # Define a threshold for similarity to determine if the images match
                    threshold = 1  # Adjust as needed

                    # Determine if the images match based on the threshold
                    if similarity_score > threshold:
                        class_num= 1
                        # print("Images match!")
                    else:
                        class_num= 0
                        # print("Images do not match.")    
                    list_fetures_data.append([features, class_num]) 
    #                 print(training_data)
                except Exception as e:
                    print("problem")
                    break    
    return list_fetures_data            

# Define the database directory
training_data = readimages("E:/4th year/MMU-Iris-Database")
print("Done reading train images")
print(len(training_data))

# print(img_array.shape[1])

#separating image's array and labels of them
X = []
y = []


for features,label in training_data:
    X.append(features)
    y.append(label)

print(y)


# # Define the database directory
# testing_data = readimages("E:/4th year/Biometrics/data/test")
# print("Done reading test images")
# print(len(testing_data))
# #separating image's array and labels of them
# X_test = []
# y_test = []

# for features,label in training_data:
#     X_test.append(features)
#     y_test.append(label)

# print(y_test)


# # Split the data into training and testing sets
# # You may also consider cross-validation for model evaluation
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



# Test the model on an input image
input_image = cv2.imread('E:/4th year/Biometrics/data/test/8/S5008L03.jpg')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

input_features = extract_eye_features(input_image)

predicted_class = model.predict([input_features])[0]

print(predicted_class)


