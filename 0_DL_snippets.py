# Chollet Deep Learning code snippets

from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras import models
from keras import layers


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


print(f"Shape train_image ={train_images.shape}")
print(f"Shape train_labels ={train_labels.shape}")
print(f"Shape train_image ={test_images.shape}")
print(f"Shape train_image ={test_labels.shape}")

network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))

network.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

