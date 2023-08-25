from Weight_Mapping import Crossbar_Map, SimpleSLP
from tensorflow.keras.datasets import mnist
from process_Images import get_Data
import os
import numpy as np
import csv
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




def load_and_store_info():
    # Define the classes and initialize the list to store dictionaries
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    con_dicts = []

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if i != j:
                cls_str = str(i) + 'vs' + str(j) + '_'
                print("Processing classes: {}".format(cls_str))

                # Load the previously saved weights and conductances
                slp_W_filename = 'ovo_weights/slp_W_{}.npy'.format(cls_str)
                G_filename = 'ovo_weights/G_{}.npy'.format(cls_str)
                
                slp_W = np.load(slp_W_filename)
                G = np.load(G_filename)

                # dictionary for specific class combination
                conductance_info = {
                    'classes': cls_str,
                    'G': G,
                    'weights': slp_W,
                    'label_0': i,
                    'label_1': j
                }

                con_dicts.append(conductance_info)

    # Save the list of dictionaries
    with open('ovo_weights/con_dicts.pkl', 'wb') as f:
        pickle.dump(con_dicts, f)

    print("Data loaded and stored successfully!")



def test_RRAM(cls_str,known_conductances,x_train,y_train,x_test,y_test):
    print("Testing RRAM...")
    print("Classes: {}".format(cls_str))
            

    # Create an instance of SimpleSLP with the appropriate weight dimensions
    slp = SimpleSLP(known_conductances,4,w_dim=(28*28, 1))
    # Hyperparameters
    learning_rate = 0.01
    epochs = 25
    batch_size = 1
    
    # Flatten the training and testing data from (28, 28) to (784,)
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)

    # Train the SimpleSLP
    slp.train(x_train, y_train, learning_rate, epochs, batch_size)

    # Test the SimpleSLP
    slp.test(x_test, y_test)


    crossbar = Crossbar_Map(known_conductances,[slp.W],slp.bias,slp.W.shape)
    crossbar.compute_deltaW()
    crossbar.compute_Gmatrix()
    percentage, G, delta_w, delta_wG  = crossbar.test_Ideal(x_train,y_train)

     # Construct the filenames based on cls_str
    slp_W_filename = 'ovo_weights/slp_W_{}.npy'.format(cls_str)
    G_filename = 'ovo_weights/G_{}.npy'.format(cls_str)

    # Save the arrays with the new filenames
    np.save(slp_W_filename, slp.W)
    np.save(G_filename, G)

    return slp.W,G, delta_w, delta_wG, slp.bias

    



def ovo(known_conductances,x_train_full,y_train_full,con_dicts):

    slp_acc = []
    mpl_acc = []

   # Conduct 100 trials
    for i in range(100):
        print(f'Trial:{i}')
        
        print("Testing SLP...")
        slp = SimpleSLP(known_conductances, 0, w_dim=(28*28, 1))
        accuracy = slp.test_Weight_OVO(con_dicts, x_train_full, y_train_full)

        print("Testing crossbar...")
        crossbar = Crossbar_Map(known_conductances, w_dim=(0,0))
        prev_accuracy = crossbar.test_Ideal_OVO(con_dicts, x_train_full, y_train_full)

        print("SLP Accuracy:", accuracy)
        print("RRAM Accuracy:", prev_accuracy)

        slp_acc.append(accuracy)
        mpl_acc.append(prev_accuracy)

    # Save accuracies to a CSV file
    with open('results_data/accuracies.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trial", "SLP Accuracy", "RRAM Accuracy"])  # Writing the header
        
        for i, (slp_a, mpl_a) in enumerate(zip(slp_acc, mpl_acc)):
            writer.writerow([i + 1, slp_a, mpl_a])

    # Compute and print the average of accuracies
    avg_slp_acc = np.mean(slp_acc)
    avg_mpl_acc = np.mean(mpl_acc)

    print(f"Average SLP Accuracy: {avg_slp_acc:.2f}")
    print(f"Average MPL Accuracy: {avg_mpl_acc:.2f}")





def main():
    

    # Load the MNIST dataset outside the loops
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    
    input_size = 28*28 
    #classes = [0,1,2,3,4,5,6,7,8,9]
    classes = [0,1,2,3,4,5,6,7,8,9]
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
    
    con_dicts = []
    # Check if the file for conductance dict exists
    if os.path.exists('results_data/con_dicts.pkl'):
        # Load from file
        with open('results_data/con_dicts.pkl', 'rb') as f:
            con_dicts = pickle.load(f)
    
    else:
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

            
                        slp_w,G, delta_w, delta_wG, biases  = test_RRAM(cls_str,known_conductances,x_train,y_train,x_test,y_test)


                        # Define the classes and initialize the list to store dictionaries
        
                        # Populate the dictionary for this class combination
                        conductance_info['classes'] = cls_str
                        conductance_info['G'] = G
                        conductance_info['delta_w'] = delta_w
                        conductance_info['delta_wG'] = delta_wG
                        conductance_info['weights'] = slp_w
                        conductance_info['biases'] = biases
                        conductance_info['label_0'] = i
                        conductance_info['label_1'] = j


                        con_dicts.append(conductance_info)

        # Save the list of dictionaries for later use
        with open('ovo_weights/con_dicts.pkl', 'wb') as f:
            pickle.dump(con_dicts, f)

        print("Data loaded and stored successfully!")

    ovo(known_conductances,x_train_full,y_train_full,con_dicts)
    

    


if __name__ == "__main__":
    main()