#Secure connection
import ssl

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

#Import 60,000 28 x 28 images from the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()

#Neural net structure
#Activtion functions in each layers is another layer of filter, which provides a 
#threshold value for classification
model = keras.Sequential([
    #Flattens out a a 28 x 28 matrix into input layer, 1 node for each pixel (784x1 matrix)
    #Flattening simplifies neural net structure into a 1D array column
    keras.layers.Flatten(input_shape=(28, 28)), 

    #Hidden layer of 128 nodes, relu returns values of the value or 0. If negative value
    #(from negative edge), turn to 0. If not, pass on the next value
    keras.layers.Dense(128, activation=tf.nn.relu),

    #Output layer for 10 labels (nodes). Dense connects each node of output layer to each
    #node of the hidden layer. tf.nn.softmax takes the incoming node with highest probability
    #and makes it 1, the rest are 0
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Compile the model for training. Optimizer and Loss functions:
# 1. Loss function tells how to determine the correct/incorrect level we are at
# (how far off from the correct node)
# 2. Optimizer functions tells how to modify weight values to get more correct values
# (lower or increase proper edge weights)
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Start to train. Epochs=number of times passing through the neural net
model.fit(train_imgs, train_labels, epochs=5)

#Test model with test data (how much loss is there)
test_loss = model.evaluate(test_imgs, test_labels)
print('Test loss ' + str(test_loss[0]))

#Predict with the model
predictions = model.predict(test_imgs)

#Test print with test_image 0, should be 9 for ankle boot
print(test_labels[0])
print(predictions[0])
print(list(predictions[0]).index(max(predictions[0])))
plt.imshow(test_imgs[0], cmap='gray', vmin=0, vmax=255)