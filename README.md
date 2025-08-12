# Rice Image Classification using Artificial Neural Network (ANN)

## Project Overview

This project demonstrates how to build an Artificial Neural Network (ANN) for image classification using TensorFlow and Keras. The goal is to classify different types of rice grains from images into their respective categories using an ANN model. The project covers dataset loading, preprocessing, model building, training, evaluation, and explains the ANN architecture in detail.

---

## Dataset

The dataset consists of rice grain images organized into subfolders, each representing a rice type/class:

1. Arborio
2. Basmati
3. Ipsala
4. Jasmine
5. Karacadag
   
Source: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

---

- Each folder contains images of rice grains belonging to that class.
- Images vary in size and format (.jpg, .png, etc.).
- The dataset is automatically split into training and validation subsets with an 80-20 ratio during loading.

---

## Preprocessing

1. **Loading Dataset:**  
   Uses `tf.keras.utils.image_dataset_from_directory` to load images and assign labels based on folder names.

2. **Image Resizing:**  
   All images resized to 64×64 pixels to ensure consistent input size.

3. **Normalization:**  
   Pixel values scaled from `[0, 255]` to `[0, 1]` to help the neural network train efficiently.

4. **Flattening:**  
   Since ANNs require 1D input, images are flattened from 3D `(64, 64, 3)` arrays to 1D vectors of length 12,288.

---

## ANN Model Architecture

- **Input Layer:**  
  Takes flattened vectors of size 12,288.

- **Hidden Layers:**  
  - Dense layer with 128 neurons and ReLU activation.  
  - Dense layer with 64 neurons and ReLU activation.

- **Output Layer:**  
  Dense layer with neurons equal to the number of classes, using softmax activation for classification.

- **Compilation:**  
  - Optimizer: Adam  
  - Loss: Sparse Categorical Crossentropy (for integer labels)  
  - Metrics: Accuracy

---

## Training

- Trained for 20 epochs with batch size 32.
- Uses validation split for performance monitoring.
- Dataset shuffling enabled for better generalization.

---

## Results

- Evaluation on validation data yields accuracy and loss metrics.
- Model performance shows its ability to correctly classify unseen rice grain images.
- Future improvements could include switching to CNNs for better image feature extraction.



