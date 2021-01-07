# Tensorflow-Image-Classifier

Use Keras as the library to build neural nets as keras is good for defining graph structure.

Building a neural network that classifies images after training with the Keras MNIST dataset.
A basic neural network has:
- One input layer of nodes. Each node for a pixel in the neural net.
- One hidden layer of nodes. Captures the pattern. There are often more than 1 hidden layer.
- One output layer of nodes. Each node for a result label.
keras.Sequential() defines a sequence of different columns/layers of nodes
Neural nets are fully connected between each layer's nodes.
Each edge has a weight. Wieght * Pixel is passed on to next layer. After all multiplication,
we find the proper output layer.
Done with linear algebra multiplication, optimized by numpy.

Keras output:
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot