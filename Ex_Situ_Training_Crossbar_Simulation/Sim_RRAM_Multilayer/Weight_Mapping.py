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



class Crossbar_Map():
    def __init__(self,conductance,W,bias):
        self.G_on = np.max(conductance)
        print(f'Max conductance {self.G_on}')
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
            G_layer = (self.W[i] - self.wmin[i]) * (self.G_on - self.G_off) / (self.wmax[i] - self.wmin[i]) + self.G_off
            self.G.append(G_layer)

    def compute_Gmatrix(self):
        self.G = []
        for i in range(len(self.W)):
            self.G.append((self.W[i] - self.wmin[i])/(self.delta_w[i]) + self.G_off)
        
        
    def compute_nonIdeal_Gmatrix(self, deviation_scale):
        self.G = [np.array(layer) for layer in self.G]
        self.nonIdeal_G = []
        self.deviations = []
        for layer in self.G:
            # generate an array of random numbers with the same shape as layer
            random_nums = np.random.uniform(-deviation_scale, deviation_scale, layer.shape)

            # create the nonIdeal_layer and clip its values to be within [0, G_on]
            nonIdeal_layer = np.clip(layer + random_nums, 0, self.G_on)

            self.nonIdeal_G.append(nonIdeal_layer)
            self.deviations.append(random_nums)

        

    def deviate(self, prev_deviation_scale):
        dev = 1
        print(f'Deviating from {self.total_deviation} to {self.total_deviation+1} microsiemens')
        self.total_deviation+=dev
        self.deviations = [np.array(layer) for layer in self.deviations]

        # Calculate the update values based on the conditions
        update_values = [np.where(layer > 0, layer + dev, layer - dev) for layer in self.deviations]
    
        # Add the update values to self.nonIdeal_G
        for i in range(len(self.nonIdeal_G)):
            self.nonIdeal_G[i] += update_values[i]

    
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
                
                
        








