# Advanced Image Classification with Transfer Learning and Fine-tuning

This Python script demonstrates advanced image classification using transfer learning and fine-tuning with a pre-trained convolutional neural network (CNN) model. The script utilizes the InceptionV3 model, fine-tunes it on a custom dataset, and performs image classification.

## Features

1. Data Preparation:
   * Uses a custom dataset of images for a specific classification task.
   * Preprocesses the images by resizing them to a consistent size and applying data augmentation techniques.

2. Transfer Learning:
   * Loads the pre-trained InceptionV3 model from the Keras library.
   * Initially freezes the weights of the pre-trained layers to retain the learned features.
   * Replaces the top layers of the model with custom dense layers suitable for the specific classification task.

3. Two-phase Training:
   * Initial Training:
     - Trains the model with frozen pre-trained layers.
     - Allows the custom top layers to learn features specific to the new dataset.
   * Fine-tuning:
     - Unfreezes the top layers of the pre-trained model.
     - Continues training with a lower learning rate to fine-tune the model for the specific task.

4. Model Training:
   * Splits the dataset into training and validation sets.
   * Trains the modified model on the training data using data generators and callbacks.
   * Utilizes techniques such as learning rate scheduling, early stopping, and model checkpointing to optimize training.

5. Model Evaluation:
   * Evaluates the trained model on the validation set and calculates evaluation metrics.
   * Generates a classification report and confusion matrix for detailed performance insights.

6. Inference and Visualization:
   * Allows the user to provide new images for classification using the trained model.
   * Preprocesses the input images and passes them through the model to obtain predictions.
   * Visualizes the predicted class labels and corresponding probabilities for each input image.

7. Training History Visualization:
   * Plots the training and validation accuracy/loss curves for both initial training and fine-tuning phases.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- numpy

## Usage

1. Clone the repository or download the script file.
2. Install the required dependencies using pip:
   ```
   pip install tensorflow keras scikit-learn matplotlib numpy
   ```
3. Prepare your custom dataset:
   * Organize your dataset into subdirectories, where each subdirectory represents a class and contains the corresponding images.
   * Update the `dataset_path` variable in the script with the path to your dataset directory.
4. Run the script:
   ```
   python advanced_image_classification.py
   ```
5. The script will perform the following steps:
   * Preprocess the dataset and split it into training and validation sets.
   * Load the pre-trained InceptionV3 model and prepare it for fine-tuning.
   * Perform initial training with frozen pre-trained layers.
   * Fine-tune the model by unfreezing top layers and continuing training.
   * Evaluate the model on the validation set.
   * Generate a classification report and confusion matrix.
   * Allow you to provide new images for classification and visualize the predictions.
6. View the training history visualization, which includes accuracy and loss curves for both training phases.

## Customization

- You can modify the `img_size` variable to adjust the input image size for the model.
- Experiment with different pre-trained CNN models by replacing `InceptionV3` with other models like `ResNet50` or `VGG16`.
- Adjust the hyperparameters such as batch size, number of epochs, and learning rate for both initial training and fine-tuning phases to suit your specific dataset and requirements.
- Customize the custom dense layers added on top of the pre-trained model to match your classification task.
- Modify the number of layers to unfreeze during fine-tuning by adjusting the slicing in the `model.layers[-20:]` line.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The script utilizes the InceptionV3 model, which is pre-trained on the ImageNet dataset.
- The data augmentation techniques are implemented using the ImageDataGenerator class from the Keras library.
