# Identical Twins Detection

This project uses a deep learning model based on **ResNet-50** to detect identical twins from images. Users can upload an image, and the model will compare it against a reference dataset to determine identical twin matches.

## Features
- **Deep Learning Model**: Utilizes ResNet-50 for feature extraction and twin identification.
- **Image Upload**: Users can upload an image for twin detection.
- **Dataset Matching**: The model searches a dataset for the closest match to the uploaded image.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/twin-detect.git
   cd twin-detect
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```
2. **Upload an Image:**
   - Upload an image for twin detection.
   - The model will compare the uploaded image with the dataset and display the best match.

## Model Details
- **Architecture**: ResNet-50 pre-trained on ImageNet
- **Input**: Images uploaded by the user
- **Output**: Identical twin match from the dataset

## Demo
![Image](https://github.com/user-attachments/assets/0e1d8e10-15ec-4fcb-b30e-ddae6e46de6d)

## Dataset
The dataset contains labeled twin images for training and matching purposes. Ensure the dataset is placed in the `dataset/` directory before running the model.
