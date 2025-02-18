import cv2
import numpy as np
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


# Load and preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, target_size)  # Resize for ResNet50 input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


# Extract features using ResNet50
def extract_features(image, model):
    return model.predict(image, verbose=0)


# Calculate cosine similarity
def cosine_similarity(features1, features2):
    dot_product = np.dot(features1, features2.T)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    return dot_product / (norm1 * norm2)


# Visualization
def visualize_results(input_image_path, matched_image_path, similarity_score, threshold):
    if similarity_score < threshold:
        plt.figure(figsize=(6, 6))
        plt.title("No similar image found")
        plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
        axs[0].set_title("Input Image")
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(cv2.imread(matched_image_path), cv2.COLOR_BGR2RGB))
        axs[1].set_title("Twin or Original Image Match")
        axs[1].axis('off')

        plt.suptitle(f"Similarity score: {similarity_score:.2f}")
        plt.show()


# Main program
def main():
    # Load the pre-trained ResNet50 model
    base_model = ResNet50(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)

    # Input image path
    input_image_path = 'D:/input.jpg'  # Replace with your input image path
    input_image = load_and_preprocess_image(input_image_path)
    input_features = extract_features(input_image, model)

    # Directories containing images to compare
    twin_images_dir = 'D:/Twin'  # Replace with your directory path for twin images
    original_images_dir = 'D:/Original'  # Replace with your directory path for original images

    # Similarity threshold
    similarity_threshold = 0.8

    # Find the most similar image from both directories
    most_similar_image = None
    highest_similarity_score = -1

    # Compare against twin images
    for file_name in os.listdir(twin_images_dir):
        twin_image_path = os.path.join(twin_images_dir, file_name)
        twin_image = load_and_preprocess_image(twin_image_path)
        twin_features = extract_features(twin_image, model)

        similarity = cosine_similarity(input_features, twin_features)
        if similarity > highest_similarity_score:
            highest_similarity_score = similarity
            most_similar_image = twin_image_path

    # Compare against original images
    for file_name in os.listdir(original_images_dir):
        original_image_path = os.path.join(original_images_dir, file_name)
        original_image = load_and_preprocess_image(original_image_path)
        original_features = extract_features(original_image, model)

        similarity = cosine_similarity(input_features, original_features)
        if similarity > highest_similarity_score:
            highest_similarity_score = similarity
            most_similar_image = original_image_path

    # Display results
    if highest_similarity_score < similarity_threshold:
        print("No similar image found")
        visualize_results(input_image_path, None, highest_similarity_score, similarity_threshold)
    else:
        # Extract the scalar value from the numpy array
        highest_similarity_score_value = highest_similarity_score[0][0]
        
        print(f"Most likely image found: {most_similar_image}")
        print(f"Similarity score: {highest_similarity_score_value}")
        visualize_results(input_image_path, most_similar_image, highest_similarity_score_value, similarity_threshold)


# Run the program
if __name__ == "_main_":
    main()