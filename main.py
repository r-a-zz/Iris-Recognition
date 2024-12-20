import cv2
import os
import time
import sys
from utils import preprocess, find_best_match, select_image_from_dataset, display_images, show_details
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to display confusion matrix
def display_confusion_matrix():
    # Example data (predicted vs actual)
    # Rough data: 5 data points (predicted labels vs actual labels)
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 1, 1, 2, 2]
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plotting confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Main function
if __name__ == '__main__':
    dataset_folder = r'..\Iris-Recognition\dataset'
    
    # Switch case for user choice
    choice = input("Choose an option: \n1. Iris Recognition\n2. Display Confusion Matrix\nYour choice: ").strip()

    switch = {
        '1': lambda: print("Iris recognition process starting..."),
        '2': display_confusion_matrix,
    }

    # Execute the chosen option
    switch.get(choice, lambda: print("Invalid choice. Proceeding with Iris Recognition process..."))()

    if choice == '1':
        print("Please select an image from the dataset for matching:")
        query_image, individual_id, session_id = select_image_from_dataset(dataset_folder)
        
        if query_image is not None:
            start_time = time.time()  # Start measuring time
            
            best_match_id, similarity = find_best_match(dataset_folder, query_image)
            
            end_time = time.time()  # Stop measuring time
            execution_time = end_time - start_time

            # Load and display the input image
            query_images = [query_image]
            print("Displaying input image:")
            display_images(query_images)
            
            print("Best match found:")
            print(f"Individual ID: {best_match_id}")
            print(f"Similarity: {similarity}")
            print(f"Execution Time: {execution_time} seconds")
            
            # Display segmentation and normalization process details for the input image
            print("Displaying segmentation and normalization process details for the input image:")
            show_details(query_image)

            if best_match_id is not None:
                print(f"Displaying all images of the matched individual {best_match_id}:")
                individual_images = []
                for filename in os.listdir(dataset_folder):
                    if filename.startswith(best_match_id) and filename.endswith(".jpg"):
                        filepath = os.path.join(dataset_folder, filename)
                        img = cv2.imread(filepath)
                        individual_images.append(img)
                display_images(individual_images)

            # Close all OpenCV windows after processing
            cv2.destroyAllWindows()
    elif choice == '2':
        # Confusion matrix will already be displayed by display_confusion_matrix function
        pass
    else:
        print("Invalid choice. Proceeding with Iris Recognition process.")
