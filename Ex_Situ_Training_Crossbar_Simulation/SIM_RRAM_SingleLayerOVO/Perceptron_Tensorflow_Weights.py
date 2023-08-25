#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})


# # Perceptron for Binary Classification

# ## One-vs-One Classification: <br>
# 
# One-VS-One consists in dividing the problem into as many binary problems as all the possible combinations between pairs of classes, so one classifier is learned to discriminate between each pair, and then the outputs of these base classifiers are combined in order to predict the output class.
# 
# **Reference:**
# 
# *Mikel Galar, Alberto Fernández, Edurne Barrenechea, Humberto Bustince, Francisco Herrera,
# An overview of ensemble methods for binary classifiers in multi-class problems: Experimental study on one-vs-one and one-vs-all schemes,
# Pattern Recognition,
# Volume 44, Issue 8,
# 2011,
# Pages 1761-1776,
# ISSN 0031-3203,
# https://doi.org/10.1016/j.patcog.2011.01.017.*
# 
# 
# <br>
# The One-vs-One (OvO) classification method has also regularly been used for training particular machine learning algorithms such as support vector machines [10–12] or other classifiers [13]. In the OvO scheme, each binary classifier is trained to discriminate between examples of one class and examples belonging to one other class. Therefore, if there are K classes, the OvO scheme requires training and storing K(K − 1)/2 different binary classifiers, which can be seen as a disadvantage when K is large. The authors in [14] described several methods to cope with a large set of base learners for OvO. Furthermore, different algorithms have been proposed to improve the OvO scheme [15,16]. An advantage of the OvO scheme is that the datasets of individual classifiers are balanced when the entire dataset is balanced. Comparisons between using the OvO scheme and the OvA scheme have shown that OvO is better for training support vector machines [10,17] and several other classifiers [13].
# 
# 
# **Reference:**
# 
# *Pornntiwa Pawara, Emmanuel Okafor, Marc Groefsema, Sheng He, Lambert R.B. Schomaker, Marco A. Wiering,
# One-vs-One classification for deep neural networks,
# Pattern Recognition,
# Volume 108,
# 2020,
# 107528,
# ISSN 0031-3203,
# https://doi.org/10.1016/j.patcog.2020.107528.*
# 
# 
# <br>
# The One-vs-One strategy is one of the most commonly used decomposition technique to overcome multi-class classification problems; this way, multi-class problems are divided into easier-to-solve binary classification problems considering pairs of classes from the original problem, which are then learned by independent base classifiers.
# 
# **Reference:**
# 
# *Dynamic classifier selection for One-vs-One strategy: Avoiding
# non-competent classifiers
# Mikel Galar*

# ### Specify the digits to classify



class Perceptron:
    def __init__(self, input_size):
        self.input_size = input_size


    #Create the model and specify activation function
    def create_model(self,input_size):
        model = Sequential()
        model.add(Dense(1, input_dim=input_size, activation='sigmoid')) 
        return model


    def extract_Weights(self,x_train, y_train, x_test, y_test,cls_str):
        
        weights = []
        biases = []
        i = 0

        # Load weights and biases if they exist
        while True:
            weight_path = f'Weights/{cls_str}weights_layer_{i}.npy'
            bias_path = f'Weights/{cls_str}biases_layer_{i}.npy'

            try:
                print("Loading:{}".format(weight_path))
                weight = np.load(weight_path)
                bias = np.load(bias_path)
                weights.append(weight)
                biases.append(bias)
                i += 1
            except FileNotFoundError:
                break
        

        if len(weights) == 0:
            # Create a perceptron instance
            slp = self.create_model(self.input_size)

            # Compile the model, specify optimizer and loss function
            slp.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])


            # Convert input_vectors and binary_labels into NumPy arrays
            input_vectors = np.array(x_train, dtype=np.float32)
            y_labels = np.array(y_train, dtype=np.float32)

            # Flatten input_vectors
            input_vectors = input_vectors.reshape(-1, self.input_size)

            # Train the perceptron
            history = slp.fit(input_vectors, y_labels, epochs=50, batch_size=32, verbose=1)
            
            # Plot the model
            plot_model(slp, to_file='Figures/model.png', show_shapes=True, show_layer_names=True, dpi=300)
            print(slp.summary())
            
            # print the final epoch's metrics
            #print("Final epoch's loss:", history.history['loss'][-1])
            #print("Final epoch's accuracy:", history.history['accuracy'][-1])

            # Get the weights and biases
            weights_and_biases = slp.get_weights()

            # Save each layer's weights and biases to separate files
            # Ensure 'Weight' directory exists
            if not os.path.exists('Weights'):
                os.makedirs('Weights')


            for k in range(len(weights_and_biases) // 2):
                np.save(f'Weights/{cls_str}weights_layer_{k}.npy', weights_and_biases[k * 2])
                np.save(f'Weights/{cls_str}biases_layer_{k}.npy', weights_and_biases[k * 2 + 1])
                weights.append(weights_and_biases[k * 2])
                biases.append(weights_and_biases[k * 2 + 1])



            X_test = np.array(x_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)

            # Flatten X_test
            X_test = X_test.reshape(-1, self.input_size)

            loss, accuracy = slp.evaluate(X_test, y_test)
            print("Test set loss:", loss)
            print("Test set accuracy:", accuracy)


            new_image = x_test[1]
            new_image = np.array(new_image, dtype=np.float32).reshape(1, -1)
            prediction = slp.predict(new_image)
            print(prediction)
            print("_______________________________")

        return weights, biases








