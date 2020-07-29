import numpy as np
from Load_Data import train_dir
from Load_Data import Image_to_array
import PIL
from PIL import Image
import matplotlib.pyplot as plt

def generate_background(image_shape=(512, 512), *, max_difference=0.1):
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


def make_plate(x_train, y_train):
    """
    Makes "plates" by scaling down pictures
    Needs to create a backround and put plates in so that they don't overlap
    
    Parameters
    ----------
    x_train : images
        The names of the image files

    Y_train : labels
        The lables of the images 
    
    Returns
    -------
    Plate
        one Pillow with 1-3 pic on it

    Labels: int
        int corresponding to class

    Boxes : tuple[left, top, right, bottom] (ints)
        coord of the boxes
        
    """
    #generate backround
    food_amount = np.random.randint(1, 4)

    #backround array
    backround = generate_background()
    #backround pillow
    backround_pillow = Image.fromarray(np.uint8(backround)).convert('RGB') #np.unit8 converts to integers, convert('RGB') converts to RGB


    x_pics_list = []
    x_pics_string = []
    y_pics_list = []

    for i in range(food_amount):
            index = np.random.randint(0, len(x_train))
            
            x_pic = x_train[index]
            x_pic_array = Image_to_array(train_dir + "\\" + x_pic)
            
            y_pic = y_train[index]
            
            
            x_pics_list.append(x_pic_array)
            x_pics_string.append(x_pic)

            y_pics_list.append(y_pic)


    boxes = []
    coordxmin = []
    coordxmax = []
    coordymin = []
    coordymax = []
    for i in range(len(x_pics_string)):

        im = Image.open(train_dir + "\\" + x_pics_string[i])

        maxsize = np.random.randint(100, 300)
                    
        cood_x = np.random.randint(0, (512-maxsize))
        cood_y = np.random.randint(0, (512-maxsize))

        mindist = 2*maxsize/3

        coordxmax.append(cood_x+ mindist)
        coordxmin.append(cood_x-mindist)
        coordymax.append(cood_y+mindist)
        coordymin.append(cood_y-mindist)
        
        box = cood_x, cood_x, cood_x + maxsize, cood_y + maxsize # left, top, right, bottom
        
        boxes.append(box)
        
        maxsize = (maxsize, maxsize) #max size of scaled down image


        im.thumbnail(maxsize, PIL.Image.ANTIALIAS) #makes im into scaled down PIL

        backround_pillow.paste(im, coord) #paste image into backround PIL


    return backround_pillow, y_pics_list, boxes

def generate_plate_set(base_path, num_images, x_train, y_train):
    labels, boxes = [], []
    for i in range(num_images):
        
        img, lbls, bxs = make_plate(x_train, y_train)
        
        plt.imsave(f'{base_path}\\images\\{i:06d}.png', img)
        labels.append(lbls)
        boxes.append(bxs)
    np.save(f'{base_path}\\labels.npy', labels)
    np.save(f'{base_path}\\boxes.npy', boxes)

def get_coord(coordxmin, coordxmax, coordymin, coordymax):
