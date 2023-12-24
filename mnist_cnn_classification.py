# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#1. Loading Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#2. Loading dataset
train_image, train_label = mnist.load_data()[0]
test_image, test_label = mnist.load_data()[1]

#function for plotting overall images
def plot_images(images, indices, title_prefix="Image"):
    num_images = len(indices)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i, index in enumerate(indices):
        ax = axes[i]
        ax.imshow(images[index], cmap='gray')
        ax.set_title(f"{title_prefix} {index}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

#random index for plotting images
indices = [0, 5, 10, 15, 100, 1000, 10000] 
plot_images(train_image, indices, title_prefix="Image index")

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#3. pre processing
train_image = train_image.reshape((60000, 28, 28)).astype('float32')/255
test_image = test_image.reshape((10000, 28, 28)).astype('float32')/255

#converting into categorical value
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#4. Creating model
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
history = cnn_model.fit(train_image, train_label, epochs=2, batch_size=64, validation_split=0.2, verbose=1, 
                        validation_data=(test_image, test_label))
# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#5. Model summary
cnn_model.summary()

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#6. Loss & Accuracy plot
#training and validation loss values from the history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

#accuracy values from the history
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

test_loss, test_acc = cnn_model.evaluate(test_image, test_label)
train_loss, train_acc = cnn_model.evaluate(train_image, train_label)

print(f'Test Accuracy: {test_acc* 100 :.4f}')
print(f'Train Accuracy: {train_acc* 100 :.4f}')

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

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#7. Predictions
#7.1 Making predictions using your trained model on the first ten images in our test dataset and comparing those predictions to the true labels.

test_predictions = cnn_model.predict(test_image)
test_predicted_classes = np.argmax(test_predictions, axis=1)

train_predictions = cnn_model.predict(train_image)
train_predicted_classes = np.argmax(train_predictions, axis=1)
for i in range(10):
    print(f"{i + 1}: Predicted Class {test_predicted_classes[i]}, True Label {np.argmax(test_label[i])}")

for i in range(10):
    print(f"{i + 1}: Predicted Class {train_predicted_classes[i]}, True Label {np.argmax(train_label[i])}")
    
# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#7.2 Images along with their true and predicted labels
class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]

#creating function to plot images labels
def images_labels(images, true_labels, predicted_labels, class_labels, num_images=15):
    #rows based on num_images
    num_rows = (num_images + 1) // 5
     #maximum 5 columns
    num_cols = min(num_images, 5) 
    plt.figure(figsize=(12, 6))
    #for loop to iterate over number of images
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        #true class index
        true_class_index = true_labels[i]
        #predicted class index
        predicted_class_index = predicted_labels[i]
        plt.title(f"True: {class_labels[true_class_index]}\nPredicted: {class_labels[predicted_class_index]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


#train dataset correctly predicted classes 
images_labels(train_image, [np.argmax(label) for label in train_label], train_predicted_classes, class_labels, num_images=15)

#test dataset correctly predicted classes 
images_labels(test_image, [np.argmax(label) for label in test_label], test_predicted_classes, class_labels, num_images=15)

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#7.3 Displaying a subset of 10 images starting from index 100 of the train & test image array along with their true and predicted labels and image indices.

def images_labels(images, true_labels, predicted_labels, class_labels, start_index=0, num_images=15):
    #number of rows based on num_images
    num_rows = (num_images + 1) // 5 
    #5 columns
    num_cols = min(num_images, 5) 
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        true_class_index = true_labels[i]
        predicted_class_index = predicted_labels[i]
        plt.title(f"Index: {start_index + i}\nTrue: {class_labels[true_class_index]}\nPredicted: {class_labels[predicted_class_index]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
images_labels(train_image[1000:1010], [np.argmax(label) for label in train_label[1000:1010]], train_predicted_classes[1000:1010], class_labels, start_index=100, num_images=10)

images_labels(test_image[3000:3010], [np.argmax(label) for label in test_label[3000:3010]], test_predicted_classes[3000:3010], class_labels, start_index=100, num_images=10)

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#8 Cofusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tabulate import tabulate

def train_test_confusion_matrices(train_confusion, test_confusion, class_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    #confusion matrix for train data
    disp_train = ConfusionMatrixDisplay(train_confusion, display_labels=class_labels)
    disp_train.plot(ax=ax1, values_format='d', xticks_rotation='vertical')
    ax1.set_title('Train Confusion Matrix')

    #confusion matrix for test data
    disp_test = ConfusionMatrixDisplay(test_confusion, display_labels=class_labels)
    disp_test.plot(ax=ax2, values_format='d', xticks_rotation='vertical')
    ax2.set_title('Test Confusion Matrix')

    plt.tight_layout()
    plt.show()

train_confusion = confusion_matrix([np.argmax(label) for label in train_label], train_predicted_classes)
test_confusion = confusion_matrix([np.argmax(label) for label in test_label], test_predicted_classes)

train_test_confusion_matrices(train_confusion, test_confusion, class_labels)

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#9 Classification Report

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tabulate import tabulate

class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]

def class_accuracies(predictions, true_labels, class_labels):
    #convert predictions and labels to class indices if they are in a different format
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



def class_accuracies_table(class_accuracies, class_labels):
    data = []
    for i, label in enumerate(class_labels):
        data.append([label, f"{class_accuracies[i] * 100:.2f}%"])

    headers = ["Class", "Accuracy"]
    table = tabulate(data, headers, tablefmt="pretty")
    return table



test_class_accuracies = class_accuracies(test_predictions, test_label, class_labels)
test_accuracy_table = class_accuracies_table(test_class_accuracies, class_labels)
print(test_accuracy_table)



train_class_accuracies = class_accuracies(train_predictions, train_label, class_labels)
train_accuracy_table = class_accuracies_table(train_class_accuracies, class_labels)
print(train_accuracy_table)

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#10 Correct & Incorrect predict Classification

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

# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#11 Each Classes Accuracy

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


# - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - -- - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#12 ROC 

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


def roc_curves(model, train_data, train_labels, test_data, test_labels, class_labels):
    #one-hot encode labels to binary labels
    train_true_classes_binary = label_binarize(train_labels, classes=list(range(len(class_labels))))
    test_true_classes_binary = label_binarize(test_labels, classes=list(range(len(class_labels))))

    #model predictions for all classes at once
    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)

    #roc curve and roc area for each class
    train_fpr = {}
    train_tpr = {}
    train_roc_auc = {}
    test_fpr = {}
    test_tpr = {}
    test_roc_auc = {}

    n_classes = len(class_labels)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'blue', 'red', 'purple', 'brown', 'pink', 'gray'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for i, color in zip(range(n_classes), colors):
        #roc for training data
        train_fpr[i], train_tpr[i], _ = roc_curve(train_true_classes_binary[:, i], train_predictions[:, i])
        train_roc_auc[i] = auc(train_fpr[i], train_tpr[i])

        #roc for test data
        test_fpr[i], test_tpr[i], _ = roc_curve(test_true_classes_binary[:, i], test_predictions[:, i])
        test_roc_auc[i] = auc(test_fpr[i], test_tpr[i])

        #roc curve for training data on the left subplot (ax1)
        ax1.plot(train_fpr[i], train_tpr[i], color=color, lw=2, label=f'Training ROC curve (area = {train_roc_auc[i]:.2f}) for Class {i}')

        #roc curve for test data on the right subplot (ax2)
        ax2.plot(test_fpr[i], test_tpr[i], color=color, linestyle='--', lw=2, label=f'Test ROC curve (area = {test_roc_auc[i]:.2f}) for Class {i}')

    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5, color='grey')

    ax1.set_title('Training ROC Curves')
    ax2.set_title('Test ROC Curves')
    plt.show()

#roc_curves function
roc_curves(cnn_model, train_image, train_label, test_image, test_label, class_labels)

