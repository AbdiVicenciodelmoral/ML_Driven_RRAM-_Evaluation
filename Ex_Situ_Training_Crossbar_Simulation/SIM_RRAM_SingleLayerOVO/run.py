from Perceptron_Tensorflow_Weights import Perceptron
from Weight_Mapping import Crossbar_Map
from tensorflow.keras.datasets import mnist
from process_Images import get_Data
from Evaluate import evaluation_Plots
import os
import numpy as np
import pandas as pd
import pickle
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# This function does the initial mappning of the neural network to 
# simulated RRAM crossbars, represented by the conductance matrix
# Percentage: accuracy
# G: conductance matrix
# delta_w: scaling factor for the weight matrix of the neural network 
# in relation to the memristor crossbar conductivity values
# delta_Wg: This value represents an offset term that considers both 
# the weight matrix range and the memristor conductance range.
def test_RRAM(cls_str,known_conductances,x_train,y_train,weights, biases):
    print("Testing RRAM...")
    print("Classes: {}".format(cls_str))
    
    # Call the function to create the conductance matrix
    crossbar = Crossbar_Map(known_conductances,weights,biases)
    crossbar.compute_deltaW()
    # Construct the conductance matrix from the neural network weights
    crossbar.compute_Gmatrix()
    # Test the accuracy of image classification using the conductance
    # matrix.
    percentage, G, delta_w, delta_wG  = crossbar.test_Ideal(x_train,y_train)
    
    return percentage, G, delta_w, delta_wG


# This function tests the accuracy of classification using
# the extracted neural network weights. (Optional)
def predict(input_data, weights, biases):
    # Compute the dot product of weights and input data, then add the bias
    output = np.dot(input_data, weights) + biases
    
    # Apply the step function for each sample
    predictions = np.where(output > 0, 1, 0)
    
    return predictions


# This function is for testing individual classifiers on their respective class.
# It extracts the two classes being tested, from the full dataset, and creates
# a new dataset.
def get_test_Data(x_train_full, y_train_full, x_test_full, y_test_full, class1, class2):
    # Extract samples of class1
    train_mask_class1 = (y_train_full == class1)
    test_mask_class1 = (y_test_full == class1)
    
    # Extract samples of class2
    train_mask_class2 = (y_train_full == class2)
    test_mask_class2 = (y_test_full == class2)
    
    # Concatenate the extracted samples
    x_train = np.concatenate((x_train_full[train_mask_class1], x_train_full[train_mask_class2]))
    y_train = np.concatenate((y_train_full[train_mask_class1], y_train_full[train_mask_class2]))
    
    x_test = np.concatenate((x_test_full[test_mask_class1], x_test_full[test_mask_class2]))
    y_test = np.concatenate((y_test_full[test_mask_class1], y_test_full[test_mask_class2]))
    
    # Convert labels to binary (0 for class1 and 1 for class2)
    y_train = np.where(y_train == class1, 0, 1)
    y_test = np.where(y_test == class1, 0, 1)

    return x_train, y_train, x_test, y_test



def main():

    # Load the MNIST dataset outside the loops
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    
    input_size = 28*28 
    classes = [0,1,2,3,4,5,6,7,8,9]
    
    # Known Conductances, this will be used to determine which RRAM device has the Highest conductance
    # and will be used as the proxy RRAM device
    known_conductances = [2.98, 0.802, 5.17, 3.07, 3.80,0.740, 0.41667, 2.04, 0.02490, 0.24057, 
        0.04237, 0.380, 3.07, 2, 1.03, 0.792, 0.260, 1.65, 2.88, 2.30, 2.01, 1.73, 3.55, 4.03, 5.71,
        3.36, 0.301, 14.74, 16.75, 23.42, 0.758, 0.21557, 2.08, 0.02004, 5.74, 0.1847, 0.14308, 0.416, 
        0.73, 1.22, 0.6926, 0.47803, 1.5, 3.97, 1.3, 2.04, 1.67, 11.43, 5.007, 7.93,
        4.125923175310476, 4.606596646397641, 4.84214603912454, 1.7425549340442956, 1.6838702072844225, 
        4.572473708276178, 0.4716981132075472, 2.6288808854070824, 5.376922249704269, 4.86523304466284, 
        5.7359183205231155, 9.719117504130624, 3.2565864460872116, 4.441384823788057, 6.4687237208098844, 
        0.12690355329949238, 0.4484304932735426, 0.6711409395973155, 2.26510827217541, 2.2070670286256595, 
        4.140443855581318, 1.587780441720519, 1.9620153822005963, 3.115264797507788, 1.622086327434346, 
        1.9549195550603093, 8.48824378236143, 0.2958579881656805, 0.41841004184100417, 2.6011184809468073, 
        1.1567915230317192, 1.5629884338855893, 2.1053961302819126, 6.491398896462187, 6.6604502464366595, 
        8.400537634408602, 1.5903940201184845, 2.2374868547647284, 2.7747717750215046, 1.181655972680114, 
        1.3020748562834878, 4.048255202007934, 1.0135306339634116, 1.0640788269595012, 2.2787348464132715, 
        0.5319148936170213, 0.78125, 0.7874015748031497, 1.8826718879433693, 2.378630384624533, 2.91877061381746, 
        1.5003750937734435, 1.9067594622938315, 2.0378718097116817, 6.341154090044388, 6.250390649415588, 
        6.7235930881463055, 0.2857142857142857, 0.7092198581560284, 1.3328712712926185, 0.36231884057971014, 
        2.241800614253368, 2.3571563266075803, 0.11173184357541899, 0.15060240963855423, 0.20920502092050208, 
        4.28412304001371, 5.326515393629487, 5.3273666826487664, 4.203976962206247, 5.2803886366036545, 
        5.063547521393488, 0.06631299734748011, 0.09259259259259259, 0.1388888888888889]

    Ideal_acc = {}
    stats = []
    con_dicts = []


    # Check if the file for conductance dict exists
    if os.path.exists('results_data/con_dicts.pkl'):
        # Load from file
        with open('results_data/con_dicts.pkl', 'rb') as f:
            con_dicts = pickle.load(f)
        
    else:
        #Iterate through each combination of image classes
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                    if i != j:
                        conductance_info = {}
                        print("_______________________________")
                        
                        cls_str = str(i)+'vs'+str(j)+'_'
                        print("Classes: {}".format(cls_str))
                        print("Preprocessing Shape:{}".format(x_train_full.shape))

                        # Print unique classes in y_train_full before get_Data
                        print("Unique classes in y_train_full before get_Data: {}".format(np.unique(y_train_full)))

                        x_train, y_train, x_test, y_test = get_Data(x_train_full, y_train_full,x_test_full, y_test_full,i,j)

                        # Print unique classes in y_train after get_Data
                        print("Unique classes in y_train after get_Data: {}".format(np.unique(y_train)))
                        p = Perceptron(input_size)
                        weights, biases = p.extract_Weights(x_train, y_train, x_test, y_test,cls_str)
                        

                        # Making predictions on test data
                        x_test = x_test.reshape((x_test.shape[0], -1))
                        test_predictions = predict(x_test, weights, biases)
                        
                        # Calculate accuracy on test data
                        accuracy = np.mean(test_predictions == y_test) * 100
                        
                        print(f"Accuracy for class {i} vs class {j}: {accuracy}%")
                    
                        # Call the function to perform the mapping algorithm for the conductance matrix
                        # The returned variables will be put into a dict and append to a list of dicts.
                        percentage, G, delta_w, delta_wG = test_RRAM(cls_str,known_conductances,x_train,y_train,weights,biases)
                        conductance_info['classes'] = cls_str
                        conductance_info['G'] = G
                        conductance_info['delta_w'] = delta_w
                        conductance_info['delta_wG'] = delta_wG
                        conductance_info['weights'] = weights
                        conductance_info['biases'] = biases
                        conductance_info['label_0'] = i
                        conductance_info['label_1'] = j

                        Ideal_acc[cls_str[:-1]] = percentage
                        
                        modes, means = evaluation_Plots(cls_str)
                    
                        stats_instance = {'Pair': cls_str,
                                'Success Mean': means["Success Deviation"],
                                'Failure Mean': means["Failure Deviation"],
                                'Success Mode': modes["Success Deviation"],
                                'Failure Mode': modes["Failure Deviation"]
                                }
                        stats.append(stats_instance)
                        con_dicts.append(conductance_info)

        # Save to file
        with open('results_data/con_dicts.pkl', 'wb') as f:
            pickle.dump(con_dicts, f)


    
    #Once training is done we can perform the One-vs-One Strategy to make predictions
    # Convert dictionary to DataFrame
    crossbar = Crossbar_Map(known_conductances)
    prev_accuracy = crossbar.test_Ideal_OVO(con_dicts,x_train_full,y_train_full)

    succ_results = {"Deviation":[],"Accuracy":[]}
    fail_results = {"Deviation":[],"Accuracy":[]}
    
    for i in range(100):
        # Create a deepcopy of the dictionary to preserve the original
        con_dicts_copy = copy.deepcopy(con_dicts)
        print(f'Iteration {i}:')
        NonIdeal_crossbar = Crossbar_Map(known_conductances)
        succ_deviation, succ_accuracy, fail_deviation, fail_accuracy = NonIdeal_crossbar.test_nonIdeal_OVO(con_dicts_copy, x_train_full, y_train_full, prev_accuracy)
        fail_results["Deviation"].append(fail_deviation)
        fail_results["Accuracy"].append(fail_accuracy)
        succ_results["Deviation"].append(succ_deviation)
        succ_results["Accuracy"].append(succ_accuracy)
        print(f'Iteration: {i} for OVO')
        print(f'Sucess Deviation: {succ_deviation}, Success Accuracy: {succ_accuracy}')
        print(f'Fail Deviation: {fail_deviation}, Fail Accuracy: {fail_accuracy}')
        print('*')
                        
    # Create dataframes from the dictionaries
    df_success = pd.DataFrame(succ_results)
    df_failure = pd.DataFrame(fail_results)

    # Rename the columns
    df_success.columns = ['Success Deviation', 'Success Accuracy']
    df_failure.columns = ['Failure Deviation', 'Failure Accuracy']

    # Concatenate the dataframes horizontally
    df = pd.concat([df_success, df_failure], axis=1)

    # Save to csv
    if not os.path.exists('results_data'):
        os.makedirs('results_data')
    df.to_csv(f'results_data/OVO_results.csv', index=False)



    ideal_df = pd.DataFrame(list(Ideal_acc.items()),columns = ['Pair','Accuracy'])
    stats_df = pd.DataFrame(stats)
    
    # Save DataFrames to CSV file
    ideal_df.to_csv("results_data/IdealAccuracy.csv", index=False)  
    stats_df.to_csv('results_data/OVO_stats.csv',index=False)

if __name__ == "__main__":
    main()