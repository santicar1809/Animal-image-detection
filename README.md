# Cat and Dog Classifier

This project implements a deep learning model to classify images of cats and dogs using TensorFlow and Keras. The dataset is preprocessed, augmented, and used to train a convolutional neural network (CNN). The final model is capable of predicting whether a given image contains a cat or a dog.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7. [Usage](#usage)
8. [Future Improvements](#future-improvements)
9. [Contributing](#contributing)

---

## Project Overview
The Cat and Dog Classifier is a binary image classification project that aims to:
- Preprocess and augment image data.
- Train a CNN model to classify images as either "Cat" or "Dog."
- Evaluate the model’s performance using metrics like accuracy, precision, and recall.

---

## Features
- **Custom Neural Network**: A sequential CNN model built from scratch.
- **Data Preprocessing**: Includes resizing, normalization, and augmentation.
- **Evaluation Metrics**: Reports precision, recall, and binary accuracy.
- **Transfer Learning**: (Optional) Use pre-trained models like MobileNetV2 for better performance.
- **Visualization**: Generates loss and accuracy plots for training and validation phases.

---

## Setup and Installation

### Prerequisites
Ensure you have Python installed (3.7+ recommended) along with the following libraries:
- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- Pandas
- Pillow

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cat-dog-classifier.git
   cd cat-dog-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your dataset in the `data/` directory. Ensure it has the following structure:
   ```
   data/
     cat/
       cat1.jpg
       cat2.jpg
       ...
     dog/
       dog1.jpg
       dog2.jpg
       ...
   ```

---

## Data Preparation
1. **Load Dataset**:
   The `load_dataset` function loads images from the `data/` directory.

2. **Preprocessing**:
   - Resize all images to 256x256 pixels.
   - Normalize pixel values to a [0, 1] range by dividing by 255.
   - Use data augmentation to enhance model generalization.

3. **Dataset Splitting**:
   - Training set: 70%
   - Validation set: 20%
   - Test set: 10%

---

## Model Architecture
The CNN model includes:
- **Convolutional Layers**: Extract spatial features using kernels of size (3, 3).
- **MaxPooling Layers**: Downsample feature maps to reduce dimensionality.
- **Dropout Layers**: Prevent overfitting by randomly dropping neurons.
- **Dense Layers**: Fully connected layers for classification.

Example architecture:
```python
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Dropout(0.2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

---

## Training and Evaluation
- **Training**:
   - Optimizer: Adam (learning rate = 0.001)
   - Loss Function: Binary Crossentropy
   - Epochs: 20 (default)
   - Validation data used for model tuning.

- **Evaluation**:
   - Precision, Recall, and Binary Accuracy calculated on the test set.

- **Visualization**:
   Training and validation loss and accuracy are plotted and saved in `./files/figs/`.

---

## Usage
1. Train the model:
   ```bash
   python train.py
   ```
2. Test the model:
   Add your test images (`cat_test.jpg` and `dog_test.jpg`) to the project directory, then run:
   ```bash
   python test.py
   ```

3. Load a saved model and make predictions:
   ```python
   from tensorflow.keras.models import load_model

   model = load_model('./models/catdogmodel.h5')
   img = cv2.imread('path_to_image.jpg')
   resized_img = tf.image.resize(img, (256, 256)) / 255.0
   prediction = model.predict(np.expand_dims(resized_img, axis=0))
   print('Prediction:', 'Dog' if prediction > 0.5 else 'Cat')
   ```

---

Loss and accuracy plots are saved in `./files/figs/`.

---

## Future Improvements
- Use more diverse datasets to improve generalization.
- Experiment with transfer learning using ResNet, Inception, or EfficientNet.
- Implement hyperparameter tuning with libraries like Keras Tuner or Optuna.
- Deploy the model as a web app using Flask or FastAPI.

---

## Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!

---
