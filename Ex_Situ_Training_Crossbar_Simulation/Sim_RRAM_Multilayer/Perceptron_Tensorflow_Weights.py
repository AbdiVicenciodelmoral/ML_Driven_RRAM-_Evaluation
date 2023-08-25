import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import plot_model
import os


def get_Weights(weight_path_template, bias_path_template, train_images, train_labels, test_images, test_labels):
    weights = []
    biases = []
    i = 1

    while True:
        weight_path = weight_path_template.format(i)
        bias_path = bias_path_template.format(i)
        try:
            print("Loading:{}".format(weight_path))
            weight = np.load(weight_path)
            bias = np.load(bias_path)
            weights.append(weight)
            biases.append(bias)
            i += 1
        except FileNotFoundError:
            break
    print(f'Amount of layers {len(weights)}')
    if len(weights) == 0:
        print("Weights and biases files not found, executing Model...")
        # Create model
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        # Compile model
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # Train model
        model.fit(train_images, train_labels, epochs=5)


        # Plot the model
        plot_model(model, to_file='Figures/model.png', show_shapes=True, show_layer_names=True, dpi=300)
        print(model.summary())
        # Get and save weights of each layer
        for i, layer in enumerate(model.layers):
            layer_weights = layer.get_weights()  # list of numpy arrays
            if layer_weights:  # check if layer_weights is not empty
                weight, bias = layer_weights
                weights.append(weight)  # append to weights list
                biases.append(bias)  # append to biases list
                weight_path = weight_path_template.format(i)
                bias_path = bias_path_template.format(i)
                np.save(weight_path, weight)  # save weights to a numpy binary file
                np.save(bias_path, bias)  # save biases to a numpy binary file

        # Evaluate accuracy
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        print('\nTest accuracy:', test_acc)

    # Return weights either loaded from file or converted to numpy from the model
    return weights, biases