import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_image, train_label = mnist.load_data()[0]
test_image, test_label = mnist.load_data()[1]

#plotting overall images
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
    
indices = [0, 5, 10, 15, 100, 1000, 10000] 
plot_images(train_image, indices, title_prefix="Image index")

#pre processing
train_image = train_image.reshape((60000, 28, 28)).astype('float32')/255
test_image = test_image.reshape((10000, 28, 28)).astype('float32')/255

#converting into categorical value
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

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

class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]

def images_labels(images, true_labels, predicted_labels, class_labels, num_images=15):
    #rows based on num_images
    num_rows = (num_images + 1) // 5
     #maximum 5 columns
    num_cols = min(num_images, 5) 
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        true_class_index = true_labels[i]
        predicted_class_index = predicted_labels[i]
        plt.title(f"True: {class_labels[true_class_index]}\nPredicted: {class_labels[predicted_class_index]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


images_labels(train_image, [np.argmax(label) for label in train_label], predicted_classes, class_labels, num_images=15)
