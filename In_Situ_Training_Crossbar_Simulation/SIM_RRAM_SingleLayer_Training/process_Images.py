import numpy as np
import cv2



def downsample_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def get_Data(x_train_full, y_train_full,x_test_full, y_test_full,i,j,downsample=False,target_size=0):
    
    # Filter the data for digit classes
    filter_train = np.where((y_train_full == i) | (y_train_full == j))
    filter_test = np.where((y_test_full == i) | (y_test_full == j))

    x_train = x_train_full[filter_train]
    y_train = y_train_full[filter_train]
    x_test = x_test_full[filter_test]
    y_test = y_test_full[filter_test]

    # Use vectorized conversion to O and 1 labels
    y_train = np.where(y_train == j, 1, 0).astype(np.float32)
    y_test = np.where(y_test == j, 1, 0).astype(np.float32)
                    
                    
    if downsample:
        # Defined by the target size 
        # Downsample the training and test images
        x_train = np.array([downsample_image(img, target_size) for img in x_train])
        x_test = np.array([downsample_image(img, target_size) for img in x_test])

    # Normalize the pixel values (to be between 0 and 1)
    x_train = x_train.astype('float64') / 255.
    x_test = x_test.astype('float64') / 255.
    
    return x_train, y_train, x_test, y_test