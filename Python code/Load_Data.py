from pathlib import Path
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#link to dataset https://www.kaggle.com/vermaavi/food11


#for reference paths, will move eventually
path = r"C:\Users\khoui\Nutrient_Detection\Food_Data"
train_dir = r"C:\Users\khoui\Nutrient_Detection\Food_Data\food-11\training"
test_dir = r"C:\Users\khoui\Nutrient_Detection\Food_Data\food-11\validation"


def load_draw_data(path):
    """
    Get the Data from the path
    
    Parameters
    ----------
    path : string
        The path that the data is stored in
        
    Returns
    -------
    Image_names : List of 9866 strings that correspond to file names of pictures
    
    labels : List of 9866 ints that correspond to 11 classes
        
    """

    #make sure u have the paths in the right directory at the begining
    train_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    test_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for file in train_files:
        x_train.append(file)
        label= file.find("_")
        y_train.append(int(file[0:label]))
    for file in test_files:
        x_test.append(file)
        label= file.find("_")
        y_test.append(int(file[0:label]))

    return x_train, y_train, x_test, y_test

def ImagesNames_to_list(dir_path):
    """
    Turns all the images in a folder into numpy rgb arrays
    
    Parameters
    ----------
    path : string
        The path to the images
    
    Returns
    -------
    Picture list : list of len #picture of arrays of shape (height, width, 3)
        arrays of each pic in RGB
        
    """ 
    picture_list = []
    for dirname, _, filenames in os.walk(dir_path):
        for filename in filenames:
            picture = np.array(Image.open(os.path.join(dirname, filename)))
            picture_list.append(picture)

    return picture_list

def Image_to_array(path):
    """
    Turns singular image into array

    Parameters
    ----------
    path : string
        The path to the image
    
    Returns
    -------
    Picture array (lenght, width, 3)
    
    """
    picture = np.array(Image.open(path))
    return picture  

def show_from_matrix(pic):
    """
    shows image from a specific matrix

    Parameters
    ---------
    pic : np.array, (length, width, 3)
        the matrix rep of the image

    """
    plt.imshow(pic)
    plt.show()


def show_from_path(path):
    """
    shows image from a specific path

    Parameters
    ---------
    path : string
        the path to the image

    """
    img=mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()

