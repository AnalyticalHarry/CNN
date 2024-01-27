#Convolutional neural network architecture for digit classification 

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import  plot_model


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
history = cnn_model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=16,
                        verbose=1, 
                        validation_data=(X_test, y_test),
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

