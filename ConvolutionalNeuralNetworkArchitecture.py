#Convolutional neural network architecture for digit classification 
import pydot
import numpy as np
import graphviz 
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from tabulate import tabulate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import  plot_model


#dataset that is used to train machine learning models
len(mnist.load_data()[0])

#dataset that is used to test machine learning models
len(mnist.load_data()[1])
  
train_image, train_label = mnist.load_data()[0]
test_image, test_label = mnist.load_data()[1]

train_image = train_image.reshape((60000, 28, 28)).astype('float32')/255
test_image = test_image.reshape((10000, 28, 28)).astype('float32')/255

#converting into categorical value
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
  
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

#Sequential model, which represents a linear stack of layers.
cnn_model = Sequential([
    #2D convolutional layer with 32 filters, each of size (3, 3),
    #ReLU activation function. set the input shape to (28, 28, 1).
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #max-pooling layer with pool size (2, 2).
    MaxPooling2D((2, 2)),
    #2D convolutional layer with 64 filters and ReLU activation.
    Conv2D(64, (3, 3), activation='relu'),
    #another max-pooling layer with pool size (2, 2).
    MaxPooling2D((2, 2)),
    #third 2D convolutional layer with 64 filters and ReLU activation.
    Conv2D(64, (3, 3), activation='relu'),
    #flatten the output from the convolutional layers into a 1D vector.
    Flatten(),
    #fully connected (dense) layer with 64 units and ReLU activation.
    Dense(64, activation='relu'),
    Dropout(0.5),
    #another fully connected layer with 10 units and softmax activation,
    #typically used for multiclass classification to output class probabilities.
    Dense(10, activation='softmax')])

cnn_model.compile(
#optimizer: 'adam' is a popular optimization algorithm that adapts the learning rate during training.
optimizer='adam', 
#loss: 'categorical_crossentropy' is a common loss function used for multiclass classification tasks.
#it measures the difference between predicted probabilities and true labels.  
loss='categorical_crossentropy',
#metrics: We want to track and report the 'accuracy' metric during training, which measures the fraction of correctly classified.
metrics=['accuracy'])

#training model and saving history
history = cnn_model.fit(train_image, train_label, 
                        epochs=50, 
                        batch_size=16,
                        verbose=1, 
                        validation_data=(test_image, test_label),
                        callbacks=[early_stopping])

cnn_model.summary()

#training and validation loss values from the history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

#accuracy values from the history
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_accuracy) + 1), training_accuracy, label='Training Accuracy')
plt.plot(range(1, len(validation_accuracy) + 1), validation_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

test_loss, test_acc = cnn_model.evaluate(test_image, test_label)
train_loss, train_acc = cnn_model.evaluate(train_image, train_label)

print(f'Test Accuracy: {test_acc* 100 :.4f}')
print(f'Train Accuracy: {train_acc* 100 :.4f}')

test_predictions = cnn_model.predict(test_image)
test_predicted_classes = np.argmax(test_predictions, axis=1)

train_predictions = cnn_model.predict(train_image)
train_predicted_classes = np.argmax(train_predictions, axis=1)

class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]

train_confusion = confusion_matrix([np.argmax(label) for label in train_label], train_predicted_classes)
test_confusion = confusion_matrix([np.argmax(label) for label in test_label], test_predicted_classes)

print(train_confusion)
print(test_confusion)

#one-hot encoded labels back to categorical labels
test_true_classes = np.argmax(test_label, axis=1)
#classification report
test_report = classification_report(test_true_classes, test_predicted_classes, target_names=class_labels)
print(test_report)

#one-hot encoded labels back to categorical labels
train_true_classes = np.argmax(train_label, axis=1)
#classification report
train_report = classification_report(train_true_classes, train_predicted_classes, target_names=class_labels)
print(train_report)

#creating function to calculate correct and false predictions
def correct_predictions(cnn_model, train_data, train_label, test_data, test_label, class_labels):
    def calculate_predictions(data, labels):
        predictions = cnn_model.predict(data)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        confusion = confusion_matrix(true_classes, predicted_classes)

        correct_predictions_per_class = []
        false_predictions_per_class = []

        for i, label in enumerate(class_labels):
            correct_predictions = confusion[i, i]  
            false_predictions = sum(confusion[i, :]) - correct_predictions 
            correct_predictions_per_class.append(correct_predictions)
            false_predictions_per_class.append(false_predictions)

        overall_correct_predictions = sum(correct_predictions_per_class)
        overall_false_predictions = sum(false_predictions_per_class)

        return correct_predictions_per_class, false_predictions_per_class, overall_correct_predictions, overall_false_predictions

    print("Test Data Predictions:")
    test_correct, test_false, overall_correct_test, overall_false_test = calculate_predictions(test_data, test_label)

    #test data
    test_data_table = []
    for i, label in enumerate(class_labels):
        test_data_table.append([label, test_correct[i], test_false[i]])

    test_headers = ["Class", "Correct Predictions", "False Predictions"]
    test_table = tabulate(test_data_table, test_headers, tablefmt="pretty")
    print(test_table)
    print("\nOverall Correct Predictions for Test Data:", overall_correct_test)
    print("Overall False Predictions for Test Data:", overall_false_test)

    print("\nTrain Data Predictions:")
    train_correct, train_false, overall_correct_train, overall_false_train = calculate_predictions(train_data, train_label)

    #train data
    train_data_table = []
    for i, label in enumerate(class_labels):
        train_data_table.append([label, train_correct[i], train_false[i]])

    train_headers = ["Class", "Correct Predictions", "False Predictions"]
    train_table = tabulate(train_data_table, train_headers, tablefmt="pretty")
    print(train_table)
    print("\nOverall Correct Predictions for Train Data:", overall_correct_train)
    print("Overall False Predictions for Train Data:", overall_false_train)

correct_predictions(cnn_model, train_image, train_label, test_image, test_label, class_labels)

#creating function to check accuracy of each classes
def display_class_accuracies(cnn_model, train_data, train_label, test_data, test_label, class_labels):
    def class_accuracies(predictions, true_labels):
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        if true_labels.ndim > 1 and true_labels.shape[1] > 1:
            true_labels = np.argmax(true_labels, axis=1)
            
        confusion = confusion_matrix(true_labels, predictions)
        class_accuracies = []

        for i in range(len(class_labels)):
            correct_predictions = confusion[i, i]  
            total_true_instances = sum(confusion[i, :])  
            accuracy = correct_predictions / total_true_instances if total_true_instances > 0 else 0.0
            class_accuracies.append(accuracy)

        return class_accuracies

    #class accuracies for test data
    test_predictions = cnn_model.predict(test_data)
    test_predicted_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(test_label, axis=1)
    test_class_accuracies = class_accuracies(test_predicted_classes, test_true_classes)

    #class accuracies for train data
    train_predictions = cnn_model.predict(train_data)
    train_predicted_classes = np.argmax(train_predictions, axis=1)
    train_true_classes = np.argmax(train_label, axis=1)
    train_class_accuracies = class_accuracies(train_predicted_classes, train_true_classes)

    #class accuracies using tabulate
    headers = ["Class", "Test Accuracy (%)", "Train Accuracy (%)"]
    data = []
    for i, label in enumerate(class_labels):
        data.append([label, f"{test_class_accuracies[i] * 100:.2f}", f"{train_class_accuracies[i] * 100:.2f}"])

    table = tabulate(data, headers, tablefmt="pretty")
    print(table)

  display_class_accuracies(cnn_model, train_image, train_label, test_image, test_label, class_labels)
