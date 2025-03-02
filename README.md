# Gesture Recognition Project

**Objective**: Develop a gesture recognition system for a smart TV that can identify five specific gestures to allow users to control the TV without using a remote.

## Table of Contents:
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Approach](#approach)
- [Technologies/Libraries Used](#technologieslibraries-used)
- [Conclusions](#conclusions)
- [Acknowledgements](#acknowledgements)
- [Glossary](#glossary)

## Problem Statement
Imagine you are working as a data scientist at a company that manufactures smart TVs. Your task is to develop a feature that recognizes five different hand gestures, allowing users to control the TV without a remote.

The gestures are detected by a webcam mounted on the TV, and each gesture is mapped to a specific action:
- **Thumbs up**: Increase the volume
- **Thumbs down**: Decrease the volume
- **Left swipe**: Skip backward 10 seconds
- **Right swipe**: Skip forward 10 seconds
- **Stop**: Pause the movie

Each gesture is captured in a short video clip, typically lasting 2-3 seconds, and consists of 30 frames (images). The videos are recorded by different users performing one of these gestures in front of the webcam, similar to how the smart TV will be used in real life.

### Dataset
The dataset is in a zip file, which contains two folders: **train** and **val**, each with two CSV files. These folders are divided into subfolders, where each subfolder corresponds to a video of a particular gesture. Each video contains 30 frames, and the frames are of two types: 360x360 or 120x160, depending on the camera used. 

Each row in the CSV files contains:
- The name of the subfolder (video)
- The gesture label
- A numeric label (from 0 to 4) corresponding to the gesture class.

To get the dataset on your system, follow these steps:
1. Open the terminal.
2. Go to the [dataset](https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL).
3. Download and unzip the `Project_data.zip`.

## Approach
1. **Step 1**: Import the necessary libraries.
2. **Step 2**: Load and understand the data.
3. **Step 3**: Prepare the data - split into training and validation sets, and set up test data generators.
4. **Step 4**: Identify the key parameters and hyperparameters for the model.
5. **Step 5**: Build and train a Conv3D model for gesture recognition.
6. **Step 6**: Experiment with combining CNN and RNN models.
7. **Step 7**: Use transfer learning with MobileNetV2 and RNN for improved performance.
8. **Step 8**: Make predictions using the trained models.
9. **Step 9**: Draw conclusions from the results.

## Technologies/Libraries Used
- **Python**: 3.10.12
- **NumPy**: 1.26.4
- **TensorFlow**: 2.17.0
- **Keras**: 3.4.1

## Conclusions
After testing various models, **Approach 3 - Model 3 (MobileNetV2 with transfer learning)** proved to be the best choice. It offers a good balance between accuracy, loss, and computational efficiency:

- **High Accuracy and Low Loss**: The model achieved 96% accuracy on training data and 88% on validation data, with low loss values (0.12 for training and 0.21 for validation).
- **Moderate Number of Parameters**: By freezing the initial 100 layers of MobileNetV2, the model uses fewer parameters while still benefiting from transfer learning.
- **Efficient Training Time**: Global Average Pooling helps reduce the feature size, leading to faster training without sacrificing performance.

### Future Work
- Further experimentation could focus on adjusting the frame dimensions or trying different CNN architectures.
- Testing the model with larger datasets or in real-time applications could improve its robustness and practical use.

## Acknowledgements
- This project was inspired by course materials from upGrad’s curriculum.

## Glossary
- **Data Augmentation**: A technique to artificially increase the size of the training dataset by applying transformations like rotation or flipping to the original images.
- **Convolutional Neural Network (CNN)**: A deep learning algorithm commonly used for processing image data.
- **Recurrent Neural Network (RNN)**: A neural network model used for sequence data, such as video frames.
- **Transfer Learning using Imagenet**: A method of leveraging a pre-trained model (like MobileNetV2) on a large dataset (like Imagenet) to speed up training for a new task.
- **Dropout**: A regularization technique used to prevent overfitting by randomly disabling some neurons during training.
- **Learning Rate (LR)**: A hyperparameter that controls the rate at which the model learns during training.
- **Overfitting**: When a model learns too much from the training data, including noise, and performs poorly on new, unseen data.
- **Early Stopping**: A technique to stop training once the model’s performance stops improving on the validation set.
- **Cross-Entropy Loss**: A loss function commonly used for classification tasks.
- **Accuracy**: A performance metric that measures the percentage of correct predictions made by the model.
- **Batch Normalization**: A technique to normalize activations in the network, improving training speed and stability.
- **Max Pooling**: A technique used in CNNs to reduce the spatial dimensions of the input data, reducing the amount of computation.
- **Softmax**: A function that converts a vector of values into probabilities, often used in the output layer of classification models.
- **Learning Rate Scheduler (ReduceLROnPlateau)**: A technique to adjust the learning rate during training to improve model convergence.
