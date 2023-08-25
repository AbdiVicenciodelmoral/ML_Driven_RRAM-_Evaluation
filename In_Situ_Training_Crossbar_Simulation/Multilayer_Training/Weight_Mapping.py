#!/usr/bin/env python
# coding: utf-8


import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
# # Weight Mapping from ANN to RRAM

# <img src="mapping.png" width="500" height="500">

# The weight matrix scaling refers to a neural network training method ex situ when all training is implemented in software on a traditional computer. Upon the training completion the computed weights are recalculated to the crossbar memristor conductivities in accordance with the above algorithm.
# 
# **Reference:**
# 
# *M. S. Tarkov, "Mapping neural network computations onto memristor crossbar," 2015 International Siberian Conference on Control and Communications (SIBCON), Omsk, Russia, 2015, pp. 1-4, doi: 10.1109/SIBCON.2015.7147235.*

# ## Load Images, specify the digits that will be classified, and downsample

# In[48]:

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, conductance, precision):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.G_on = max(conductance)
        self.G_off = min(conductance)
        self.precision = precision

    def forward(self, x):
        # Compute the delta values for each layer
        self.delta_w1 = (np.max(self.W1) - np.min(self.W1)) / (self.G_on - self.G_off)
        self.delta_w2 = (np.max(self.W2) - np.min(self.W2)) / (self.G_on - self.G_off)

        # Compute scaled precision for each layer
        scaled_precision_w1 = self.precision * self.delta_w1
        scaled_precision_w2 = self.precision * self.delta_w2

        # Create noise_w matrices for each layer based on the scaled precision
        noise_W1 = (2 * np.random.rand(self.W1.shape[0], self.W1.shape[1]) - 1) * scaled_precision_w1
        noise_W2 = (2 * np.random.rand(self.W2.shape[0], self.W2.shape[1]) - 1) * scaled_precision_w2

        # Add noise_w matrices to their respective weight matrices for this forward pass
        noisy_W1 = self.W1 + noise_W1
        noisy_W2 = self.W2 + noise_W2

        x = x.reshape(x.shape[0], -1)
        self.z1 = np.dot(x, noisy_W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, noisy_W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    

    def relu(self, x):
        return np.maximum(0, x)

    def drelu(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        # Clip values for stability
        x = np.clip(x, -500, 500)
    
        e_x = np.exp(x - np.max(x))
        epsilon = 1e-10
        return e_x / (e_x.sum(axis=1, keepdims=True) + epsilon)


    def compute_loss(self, y_true, y_pred):
        # Cross-entropy loss
        y_true = y_true.astype(int)
        n = y_true.shape[0]
        epsilon = 1e-10
        indices = np.arange(n)  # <-- Add this line
        log_probs = -np.log(y_pred[indices, y_true] + epsilon)  # <-- Modify this line
        loss = np.sum(log_probs) / n
        return loss

    def backward(self, x, y_true):
        x = x.reshape(x.shape[0], -1)  # Ensure x is flattened
        m = x.shape[0]
        dz2 = self.a2
        dz2[range(m), y_true] -= 1
        dz2 /= m

        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.W2.T) * self.drelu(self.z1)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    
    def accuracy(self, y_true, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)

    def train(self, x_train, y_train, epochs=10, batch_size=64):
        num_batches = int(x_train.shape[0] / batch_size)
        for epoch in range(epochs):
            cumulative_loss = 0
            for batch in range(num_batches):
                x_batch = x_train[batch * batch_size: (batch + 1) * batch_size]
                y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
                
                y_batch_int = np.argmax(y_batch, axis=1)  # Convert one-hot encoded labels to integer labels

                # Forward Pass
                y_pred = self.forward(x_batch)
                    
                # Compute Loss
                loss = self.compute_loss(y_batch_int, y_pred)
                cumulative_loss += loss

                # Backward Pass
                self.backward(x_batch, y_batch_int)

            avg_loss = cumulative_loss / num_batches
            train_acc = self.accuracy(y_train, self.forward(x_train))
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Training Accuracy: {train_acc * 100:.2f}%")
        return train_acc

    def test(self, x_test, y_test):
        y_pred = self.forward(x_test)
        test_loss = self.compute_loss(np.argmax(y_test, axis=1), y_pred)
        test_acc = self.accuracy(y_test, y_pred)
        print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

    



class Crossbar_Map():
    def __init__(self,conductance,W,bias):
        self.G_on = np.max(conductance)
        self.G_off = np.min(conductance)
        self.W = W
        self.bias = bias
        self.debug = False
    
   
    def compute_deltaW(self):
        self.delta_w = []
        self.delta_wG = []
        self.wmax = []
        self.wmin = []
        for i in range(len(self.W)):
            self.wmax.append(np.max(self.W[i]))
            self.wmin.append(np.min(self.W[i]))
            self.delta_w.append((self.wmax[i] - self.wmin[i])/(self.G_on - self.G_off))
            self.delta_wG.append((self.G_on * self.wmin[i] - self.G_off * self.wmax[i])/(self.G_on - self.G_off))
         

    def compute_Gmatrix(self):
        self.G = []
        for i in range(len(self.W)):
            self.G.append((self.W[i] - self.wmin[i])/(self.delta_w[i]) + self.G_off)
            
        
    def compute_nonIdeal_Gmatrix(self, deviation_scale):
        self.G = [np.array(layer) for layer in self.G]
        self.nonIdeal_G = []
        self.deviations = []
        for layer in self.G:
            random_nums = np.random.uniform(-deviation_scale, deviation_scale, layer.shape)
            nonIdeal_layer = np.clip(layer + random_nums, 0, self.G_on)
            self.nonIdeal_G.append(nonIdeal_layer)
            self.deviations.append(random_nums)


    def deviate(self, prev_deviation_scale):
        dev = 1
        print(f'Deviating from {self.total_deviation} to {self.total_deviation+1} microsiemens')
        self.total_deviation+=dev
        self.deviations = [np.array(layer) for layer in self.deviations]
        update_values = [np.where(layer > 0, layer + dev, layer - dev) for layer in self.deviations]
        for i in range(len(self.nonIdeal_G)):
            self.nonIdeal_G[i] += update_values[i]




    def sigmoid(self,x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def softmax(self,x):
        # Subtract the max for numerical stability
        e_x = np.exp(x - np.max(x))

        # Compute the softmax
        return e_x / e_x.sum(axis=0)



    # Outputs from Ideal Conductance Matrix
    def algorithm(self,x,G_mat):
        s = np.sum(x)
        y_prime0 = np.dot(G_mat[0].T, x) + self.bias[0]
        y0 = self.delta_w[0] * y_prime0 + self.delta_wG[0] * s
        #Relu function
        y0 = np.maximum(y0, 0)
        s = np.sum(y0)
        y_prime1 = np.dot(G_mat[1].T, y0) + self.bias[1]
        y1 = self.delta_w[1] * y_prime1 + self.delta_wG[1] * s
        y1 = self.softmax(y1)
        return np.argmax(y1)
    
     
    def class_prediction(self, y_pred, threshold=0.5):
        return 1 if y_pred >= threshold else 0
                 

    def test_Ideal(self,x_train,y_train):
        correct_predictions = 0
        for x,y_true in zip(x_train,y_train):
            x_flat = np.ravel(x)
            y_pred = self.algorithm(x_flat,self.G)
            if y_true == y_pred:
                correct_predictions+=1
        total_predictions = len(y_train)
        accuracy = correct_predictions / total_predictions
        print("Ideal Accuracy: {:.2f}%".format(accuracy * 100))
            
    
    def test_nonIdeal(self, x_train, y_train):
        deviation_scale = 1
        self.total_deviation = 1
        prev_deviation_scale = 0
        prev_accuracy = 0
        correct_predictions = 0
        self.compute_nonIdeal_Gmatrix(deviation_scale)
        print("Deviation: 1 microsiemens")
        while deviation_scale < self.G_on:
            nonIdeal_G_mat =  [np.array(layer) for layer in self.nonIdeal_G]
            correct_predictions = 0
            for x, y_true in zip(x_train, y_train):
                x_flat = np.ravel(x)
                y_pred = self.algorithm(x_flat,nonIdeal_G_mat)
                if y_true == y_pred:
                    correct_predictions+=1

                    
            total_predictions = len(y_train)
            accuracy = correct_predictions / total_predictions
            print("Non-Ideal Accuracy: {:.2f}%".format(accuracy * 100))
            if accuracy < 0.8:
                return prev_deviation_scale, prev_accuracy,  self.total_deviation,accuracy
                
                
            prev_deviation_scale = self.total_deviation
            prev_accuracy = accuracy
            

            self.deviate(prev_deviation_scale)
            deviation_scale = prev_deviation_scale
        
        return prev_deviation_scale, prev_accuracy, prev_deviation_scale,accuracy
                
                
        








