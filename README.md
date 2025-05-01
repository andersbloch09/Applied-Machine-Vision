# **Applied Machine Vision**

Welcome to the **Applied Machine Vision** repository! This project is designed to provide a complete solution for a mini-project in applied machine vision, focusing on training and deploying a convolutional neural network (CNN) for screw classification.

---

## **Project Overview**
This repository contains three main components:
1. **`main.py`**: The primary script used to train the final model using all available data.
2. **`final_test.py`**: A deployment script designed to run the trained model on a Jetson device for real-time inference.
3. **`cross_validation.py`**: A script used to evaluate the dataset using K-Fold Cross-Validation to analyze differences in the dataset.

---

## **Files and Their Purpose**

### **1. `main.py`**
- **Purpose**: 
  - This is the main script for building and training the final CNN model.
  - It uses the entire dataset to train the model without splitting into validation or test sets.
  - The trained model is saved as `final_model.pth` for deployment.
- **Key Features**:
  - Preprocessing pipeline includes resizing, grayscale conversion, and normalization.
  - Uses MobileNetV2 with pre-trained weights for transfer learning.
- **Output**:
  - The trained model is saved as `final_model.pth`.

### **2. `final_test.py`**
- **Purpose**:
  - This script is designed to deploy the trained model on a Jetson device for real-time inference.
  - It loads the `final_model.pth` file and performs predictions on live camera input or test images.
- **Key Features**:
  - Optimized for deployment on Jetson devices.
  - Includes real-time image capture and preprocessing.
  - Displays predictions directly on the video feed or saves them for further analysis.

### **3. `cross_validation.py`**
- **Purpose**:
  - This script performs K-Fold Cross-Validation on the dataset to evaluate the model's performance across different splits of the data.
  - It helps identify differences in the dataset and ensures the model generalizes well.
- **Key Features**:
  - Splits the dataset into training and validation sets using K-Fold Cross-Validation.
  - Logs metrics such as accuracy, precision, recall, and confusion matrices for each fold.
  - Useful for analyzing dataset variability and model robustness.

---

## **Requirements**

### **Python Version**
- Python 3.8 or higher

### **Required Libraries**
Install the following libraries using `pip`:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn opencv-python
