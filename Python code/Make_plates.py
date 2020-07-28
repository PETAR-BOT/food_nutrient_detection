import numpy as np

def generate_background(image_shape=(64, 64), *, max_difference=0.1):
    ''' Generate a random background of the given shape that is a gray color with a maximum offset of
    `max_difference`.
    
    Parameters
    ----------
    image_shape : Tuple[int, int], optional (default=(64, 64))
        The shape of the background to generate.
        
    max_difference : Real ∈ [0, 1], optional (default=0.1)
        The maximum difference between any two components of the color vector.
    
    Extended Description
    --------------------
    A "perfect" gray can be considered <z, z, z>, where `z` is some color value and the vector represents RGB
    values. For example, black (<0, 0, 0>) and white (<1, 1, 1>) "perfect" grays. An imperfect gray is a color
    value that is very nearly perfect, with some maximum difference between RGB components. For example, the
    color <0.9, 0.85, 0.85> is a near-perfect, with a slight shift toward red. This function generates a random
    background of the provided shape that is a near-perfect gray.
    '''
    if max_difference == 0:
        return np.ones((*image_shape, 3)) * np.random.rand()
    
    color_vector = np.random.rand(3)
    while np.max(color_vector.reshape(1, 3) - color_vector.reshape(3, 1)) > max_difference:
        color_vector = np.random.rand(3) 
    return np.ones((*image_shape, 3)) * color_vector


def make_plates(images, labels):
    """
    Makes "plates" by scaling down pictures
    Needs to create a backround and put plates in so that they don't overlap
    
    Parameters
    ----------
    images : string
        The names of the image files

    labels : 
        The lables of the images 
    
    Returns
    -------
    Plates

    Labels

    Boxes
        
    """
    backround = generate_background()

    
    #take random images

    #pillow library acaling down images

    #replace backround with picture, make sure they don't overlap
