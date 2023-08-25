#!/usr/bin/env python
# coding: utf-8


import numpy as np
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



class SimpleSLP():
    def __init__(self,conductance, precision, w_dim=None):
        if w_dim:
            self.G_on = np.max(conductance)
            self.G_off = np.min(conductance)
            # Initialize weights and bias with random values
            self.W = np.random.rand(*w_dim) * np.sqrt(1. / w_dim[0])
            self.bias = np.zeros((w_dim[1],))
            self.precision = precision
            self.wmax = np.max(self.W)
            self.wmin = np.min(self.W)
        
            self.delta_w = (self.wmax - self.wmin)/(self.G_on - self.G_off)

            self.P_w = precision * self.delta_w



        else:
            raise ValueError("Please provide weight dimensions (w_dim).")
    
    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, x):
        self.delta_w = (self.wmax - self.wmin)/(self.G_on - self.G_off)
        self.P_w = self.precision * self.delta_w
        current_noise = (np.random.rand(*self.W.shape) * 2 - 1) * self.P_w
        noisy_W = self.W + current_noise
        return self.sigmoid((np.dot(x,noisy_W) + self.bias))
    
    def class_prediction(self, y_pred, threshold=0.5):
        return 1 if y_pred >= threshold else 0
        
    def train(self, x_train, y_train, lr, epochs, batch_size):
        num_samples = len(x_train)
        
        # Compute how many full batches you can have
        num_full_batches = num_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data for each epoch to ensure randomness
            permutation = np.random.permutation(num_samples)
            x_train = x_train[permutation]
            y_train = y_train[permutation]
            
            for i in range(num_full_batches):
                start_idx = i * batch_size
                batch_x = x_train[start_idx:start_idx+batch_size]
                if np.all(batch_x == 0):
                    print("Warning: Batch contains only zeros!")
                    return "ERROR"
                batch_y = y_train[start_idx:start_idx+batch_size]
                batch_y = batch_y.reshape(-1, 1)
                
                # Forward Pass
                outputs = self.forward_pass(batch_x)

                # Compute the gradient using simple backpropagation
                delta = (outputs - batch_y) * outputs * (1 - outputs)
                gradient_w = np.dot(batch_x.T, delta)
                gradient_b = np.sum(delta, axis=0)

                # Update weights and biases
                self.W -= lr * gradient_w / batch_size
                self.bias -= lr * gradient_b / batch_size

                # Calculate accuracy for the current batch
                correct_predictions = 0
                for j in range(batch_size):
                    pred = self.class_prediction(outputs[j])
                    if pred == batch_y[j]:
                        correct_predictions += 1
                accuracy = correct_predictions / batch_size

                # Print current epoch, iteration, and accuracy
                #print(f"Epoch {epoch + 1}/{epochs}, Iteration {i+1}/{num_full_batches} - Accuracy: {accuracy*100:.2f}%")

    def test(self, x_test, y_test):
        correct_predictions = 0
        for x, y_true in zip(x_test, y_test):
            y_pred = self.class_prediction(self.forward_pass(x))
            if y_true == y_pred:
                correct_predictions += 1
        
        # Calculate the total number of predictions
        total_predictions = len(y_test)

        # Calculate the accuracy
        accuracy = correct_predictions / total_predictions

        # Print the accuracy
        percentage = accuracy*100
        print("Accuracy: {:.2f}%".format(percentage))
        return percentage




    def test_Weight_OVO(self, weight_dicts, x_train, y_train):
        # initiate a list to hold all the predictions
        all_predictions = []

        for W_dict in weight_dicts:
            classes = W_dict['classes']
            print(f'Processing {classes}')
            self.W = W_dict['weights']
            self.bias = W_dict['biases']
            label_0 = W_dict['label_0']
            label_1 = W_dict['label_1']

            # initiate a list to hold the predictions for the current classifier
            classifier_predictions = []

            for x in x_train:
                x_flat = np.ravel(x)
                y_pred_val = self.forward_pass(x_flat)
                y_pred = self.class_prediction(y_pred_val)
                if y_pred == 0:  
                    classifier_predictions.append(label_0)
                elif y_pred == 1:
                    classifier_predictions.append(label_1)
                
            all_predictions.append(classifier_predictions)

        # convert the list of lists to a 2D numpy array for easier manipulation
        all_predictions = np.array(all_predictions)
        print(all_predictions.shape)
        
        # transpose the array so that each row corresponds to an instance and each column corresponds to a classifier
        all_predictions = all_predictions.T

        print(all_predictions.shape)
        print(all_predictions)
        # perform majority voting for each example
        predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=all_predictions)

        print(predictions)

        # y_train is your ground truth labels
        y_train = np.array(y_train)

        # calculate accuracy
        accuracy = np.sum(predictions == y_train) / len(y_train)

        print(f'Accuracy: {accuracy * 100}%')

        return accuracy
    


class Crossbar_Map():
    def __init__(self,conductance,W=None,bias=None,w_dim=None, layers=None):
        # Initialize Variables min/max conductance, weights, bias
        #self.G_on = np.max(conductance)
        self.G_on = np.max(conductance)
        self.G_off = np.min(conductance)
        self.debug = True
        print("w_dim:",w_dim)
        if W is not None and bias is not None:
            self.W = W
            self.bias = bias
        elif w_dim:  # if W is None and input_size is provided
            self.W = [np.random.randn(*w_dim)]
            self.bias = [np.zeros((w_dim[1],))]
            #print(f'********************\n{self.W}\n********************\n')
        else:
            raise ValueError("Either provide weight matrix W and bias or provide input_size.")
        
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

        if self.debug:
            print("Delta W:")
            print(self.delta_w)
            

    def compute_Gmatrix(self):
        self.G = []
        for i in range(len(self.W)):
            self.G.append((self.W[i] - self.wmin[i])/(self.delta_w[i]) + self.G_off)
        
        
        if self.debug:    
            print("Conductance:")
            print(self.G,"\n")
            print(f'Length of first layer: {len(self.G[0])}')
            print(f'Length of items in first layer: {len(self.G[0][0])}')

        
    def compute_nonIdeal_Gmatrix(self, deviation_scale):
        self.G = [np.array(layer) for layer in self.G]
        self.nonIdeal_G = []
        self.deviations = []
        for layer in self.G:
            # generate an array of random numbers with the same shape as layer
            random_nums = np.random.uniform(-deviation_scale, deviation_scale, layer.shape)

            # create the nonIdeal_layer and clip its values to be within [0, G_on]
            nonIdeal_layer = np.clip(layer + random_nums, 0, self.G_on)

            # Append the results to the respective lists
            self.nonIdeal_G.append(nonIdeal_layer)
            self.deviations.append(random_nums)


    def sigmoid(self,x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))
    


    # Outputs from NonIdeal Conductance Matrix
    def algorithm(self,x,G_mat):
        s = np.sum(x)
        y_prime = np.dot(G_mat[0].T,x) + self.bias[0]
        y = self.delta_w[0] * y_prime + self.delta_wG[0] * s
        y = self.sigmoid(y)
        return self.class_prediction(y)
   
       
    def class_prediction(self, y_pred, threshold=0.5):
        return 1 if y_pred >= threshold else 0
        
             
            
    def test_Ideal(self,x_test,y_test):
        correct_predictions = 0
        for x,y_true in zip(x_test,y_test):
            x_flat = np.ravel(x)
            y_pred = self.algorithm(x_flat,self.G)
            if y_true == y_pred:
                correct_predictions+=1
        # Calculate the total number of predictions
        total_predictions = len(y_test)

        # Calculate the accuracy
        accuracy = correct_predictions / total_predictions

        # Print the accuracy
        percentage = accuracy*100
        print("Ideal Accuracy: {:.2f}%".format(percentage))
        return percentage, self.G, self.delta_w, self.delta_wG
            


                
    # Outputs from NonIdeal Conductance Matrix
    def algorithm_ovo(self,x,G_mat,dw,dwg,w,b):
        s = np.sum(x, dtype=np.float64)
        y_prime = np.dot(G_mat[0].T,x) + b[0]
        y = dw[0] * y_prime + dwg[0] * s
        y = self.sigmoid(y)
        return self.class_prediction(y)
                
                
    # Outputs from NonIdeal Conductance Matrix
    def algorithm_ovo(self,x,G_mat,dw,dwg,w,b):
        s = np.sum(x, dtype=np.float64)
        y_prime = np.dot(G_mat[0].T,x) + b[0]
        y = dw[0] * y_prime + dwg[0] * s
        y = self.sigmoid(y)
        return self.class_prediction(y)
     



    def test_Ideal_OVO(self, con_mats, x_train, y_train):
        # initiate a list to hold all the predictions
        all_predictions = []
        for G_dict in con_mats:
            classes = G_dict['classes']
            print(f'Processing {classes}')
            G = G_dict['G']
            dw = G_dict['delta_w']
            dwg = G_dict['delta_wG']
            w = G_dict['weights']
            b = G_dict['biases']
            label_0 = G_dict['label_0']
            label_1 = G_dict['label_1']
            # initiate a list to hold the predictions for the current classifier
            classifier_predictions = []

            for x in x_train:
                x_flat = np.ravel(x)
                y_pred = self.algorithm_ovo(x_flat, G, dw, dwg, w, b)
                if y_pred == 0:  
                    classifier_predictions.append(label_0)
                elif y_pred == 1:
                    classifier_predictions.append(label_1)
                
            all_predictions.append(classifier_predictions)

        # convert the list of lists to a 2D numpy array for easier manipulation
        all_predictions = np.array(all_predictions)
        print(all_predictions.shape)
        
        # transpose the array so that each row corresponds to an instance and each column corresponds to a classifier
        all_predictions = all_predictions.T

        print(all_predictions.shape)
        print(all_predictions)
        # perform majority voting for each example
        predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=all_predictions)

        print(predictions)

        # assume y_train is your ground truth labels
        y_train = np.array(y_train)  # replace with your actual labels

        # calculate accuracy
        accuracy = np.sum(predictions == y_train) / len(y_train)

        print(f'Accuracy: {accuracy * 100}%')

        return accuracy

    def deviate_OVO(self,nonIdeal_G,deviations,total_deviation,classes):
        dev = 1
        print(f'Deviating {classes} from {total_deviation-dev} to {total_deviation} microsiemens')
   
        
        # Assuming self.deviations is a list of numpy arrays
        self.deviations = [np.array(layer) for layer in deviations]

        # Calculate the update values based on the conditions
        update_values = [np.where(layer > 0, layer + dev, layer - dev) for layer in deviations]
    
        # Add the update values to self.nonIdeal_G
        for i in range(len(nonIdeal_G)):
            nonIdeal_G[i] += update_values[i]


        # The line of code below seems incorrect, please adjust it accordingly.
        # print("From:", prev_deviation_scale, " To:", max(abs(val[1]) for val in self.nonIdeal_G.values()))
        
        return nonIdeal_G, deviations


    def compute_nonIdeal_Gmatrix_OVO(self,G, deviation_scale):
        G = [np.array(layer) for layer in G]
        nonIdeal_G = []
        deviations = []
        for layer in G:
            # generate an array of random numbers with the same shape as layer
            random_nums = np.random.uniform(-deviation_scale, deviation_scale, layer.shape)

            # create the nonIdeal_layer and clip its values to be within [0, G_on]
            nonIdeal_layer = np.clip(layer + random_nums, 0, self.G_on)

            # Append the results to the respective lists
            nonIdeal_G.append(nonIdeal_layer)
            deviations.append(random_nums)

        return nonIdeal_G,deviations

    def test_nonIdeal_OVO(self, con_mats, x_train, y_train,prev_accuracy):
        # initiate a list to hold all the predictions
        deviation_scale = 1
        initial_deviation = True
        total_deviation = 1
        deviations = []
        prev_deviation_scale = 0
        while deviation_scale < self.G_on:
            all_predictions = []
            for G_dict in con_mats:
                classes = G_dict['classes']
                print(f'Processing {classes}')
                dw = G_dict['delta_w']
                dwg = G_dict['delta_wG']
                w = G_dict['weights']
                b = G_dict['biases']
                label_0 = G_dict['label_0']
                label_1 = G_dict['label_1']
                # initiate a list to hold the predictions for the current classifier
                classifier_predictions = []
                if initial_deviation == True:
                    print(f'Initial Deviations:{classes}')
                    G, deviations = self.compute_nonIdeal_Gmatrix_OVO(G_dict['G'],deviation_scale)
                    G_dict['deviations'] = deviations
                    G_dict['G'] = G
                else:
                     print(f'Deviation:{classes}')
                     G, deviations = self.deviate_OVO(G_dict['G'],G_dict['deviations'],total_deviation,classes)



                G = G_dict['G']
                for x in x_train:
                    x_flat = np.ravel(x)
                    y_pred = self.algorithm_ovo(x_flat, G, dw, dwg, w, b)
                    if y_pred == 0:  
                        classifier_predictions.append(label_0)
                    elif y_pred == 1:
                        classifier_predictions.append(label_1)
                    
                all_predictions.append(classifier_predictions)

            initial_deviation = False
            # convert the list of lists to a 2D numpy array for easier manipulation
            all_predictions = np.array(all_predictions)
            print(all_predictions.shape)
            
            # transpose the array so that each row corresponds to an instance and each column corresponds to a classifier
            all_predictions = all_predictions.T

            print(all_predictions.shape)
            print(all_predictions)
            # perform majority voting for each example
            predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=all_predictions)

            print(predictions)

            # assume y_train is your ground truth labels
            y_train = np.array(y_train)  # replace with your actual labels

            # calculate accuracy
            accuracy = np.sum(predictions == y_train) / len(y_train)

            print(f'Accuracy: {accuracy * 100}%')

            if accuracy < 0.8:
                #print("Accuracy Below 80") 
                #print("Deviation:",deviation_scale," Accuracy:",accuracy)
                #print(prev_deviation_scale,prev_accuracy)
                #true_dev = max(np.max(np.abs(array)) for array in self.deviations)
                return prev_deviation_scale, prev_accuracy,  total_deviation,accuracy
                
                
            prev_deviation_scale = total_deviation
            prev_accuracy = accuracy
           
            total_deviation+=1
            deviation_scale = total_deviation

        #print("Max")
        #print("Deviation:",deviation_scale)
        #print("Accuracy:",accuracy)
        return prev_deviation_scale, prev_accuracy, None,None

