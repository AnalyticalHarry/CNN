import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import  plot_model

#earlystopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

#sequential model for RNN
rnn_model = Sequential([
    #simple RNN layer with 64 units and 'relu' activation function.
    SimpleRNN(64, activation='relu', input_shape=(28, 28)),
    #dropout layer to prevent overfitting.
    Dropout(0.5),
    #fully connected (dense) layer with 64 units and 'relu' activation.
    Dense(64, activation='relu'),
    #fully connected layer with 10 units and 'softmax' activation.
    Dense(10, activation='softmax')
])

rnn_model.compile(
    #adam optimizer adapts the learning rate during training.
    optimizer='adam', 
    #categorical_crossentropy loss for multiclass classification tasks.
    loss='categorical_crossentropy',
    #accuracy metric to measure the fraction of correctly classified samples.
    metrics=['accuracy']
)

#training the RNN model and saving history
history = cnn_model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=16,
                        verbose=1, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
rnn_model.summary()

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
