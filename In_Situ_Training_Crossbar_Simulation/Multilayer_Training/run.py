from operator import ge
from Weight_Mapping import Crossbar_Map, MLP
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_RRAM(known_conductances,x_train,y_train,weights,biases):
    
    crossbar = Crossbar_Map(known_conductances,weights,biases)
    crossbar.compute_deltaW()
    crossbar.compute_Gmatrix()
    crossbar.test_Ideal(x_train,y_train)
    succ_results = {"Deviation":[],"Accuracy":[]}
    fail_results = {"Deviation":[],"Accuracy":[]}
    for i in range(100):
        succ_deviation, succ_accuracy, fail_deviation, fail_accuracy = crossbar.test_nonIdeal(x_train,y_train)
        fail_results["Deviation"].append(fail_deviation)
        fail_results["Accuracy"].append(fail_accuracy)
        succ_results["Deviation"].append(succ_deviation)
        succ_results["Accuracy"].append(succ_accuracy)
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
    df.to_csv(f'results_data/Multilayer_results.csv', index=False)





def main():

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    
    # Known Conductance
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

    

    test_accuracies = []
    train_accuracies = []
    for i in range(100):
        print(f'***** Iteration: {i} *****')
        model = MLP(28*28, 128, 10, 0.001,known_conductances,3)
        train_acc = model.train(x_train, y_train, epochs=50, batch_size=32)
        test_acc = model.test(x_test,y_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)


    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)

    print("Average Training Accuracy:", avg_train_accuracy)
    print("Average Testing Accuracy:", avg_test_accuracy)
    






if __name__ == "__main__":
    main()