import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


def read_images(database_directory):
    # Define the database directory
    training_data=[]
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
                    # features = extract_eye_features(img_array)
                    # print(features)
                    # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                    # training_data.append([features, class_num]) 
    #                 print(training_data)
                except Exception as e:
                    print("problem")
                    break    
    print(len(training_data))
    print(img_array.shape[1])