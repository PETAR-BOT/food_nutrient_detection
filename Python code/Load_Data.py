from pathlib import Path
import os

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

def make_plates(images, labels):
    """
    Makes "plates" by scaling down pictures
    Needs to create a backround and put plates in so that they don't overlap
    
    Parameters
    ----------
    images : string
        The images

    labels : 
        The lables of the images 
    
    Returns
    -------
    Plates

    Labels

    Boxes
        
    """


