import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
        self.conv3 = nn.Conv2d(20, 30, 3, padding=1)
        self.conv4 = nn.Conv2d(30, 40, 3, padding=1)
        
        self.classification = nn.Conv2d(40, 12, 1) # background/11 foods
        self.regression = nn.Conv2d(40, 12, 1)
        
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4,
                     self.classification, self.regression):
            nn.init.xavier_normal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.classification.bias[0], 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        
        classification = self.classification(x)
        regressions = self.regression(x)
        
        return classification, regressions


def forward_data(data):
    """
    Helper function to forward and also reshape data

    Parameters
    ----------
    data : numpy.ndarray, shape=(N, K, R, C)
        The data to forward through the model.
        
    Returns
    -------
    Tuple[torch.tensor shape=(Q, 12), torch.tensor shape=(Q, 12)]
        (classifications, regressions) at each feature map location in the image, where
        Q is the number of rows multiplied by the number of columns in the feature map.
    """
    classifications, regressions = model(torch.tensor(data).to(device))
    classifications = classifications.permute(0, 2, 3, 1).reshape(-1, 12) #change the order
    regressions = regressions.permute(0, 2, 3, 1).reshape(-1, 12) #not sure if this work with 12
    return classifications, regressions

def compute_detections(classifications, regressions):
    ''' Determine the bounding box given a set of regression predictions.
    
    Parameters
    ----------
    classifications : numpy.ndarray, shape=(N, K)
        The predicted classes at N anchor locations across K classes.
        
    regressions : numpy.ndarray, shape=(N, 12)
        The predicted regression offsets at N anchor locations.
        
    Returns
    -------
    numpy.ndarray, shape=(N, 6)
        [[left, top, right, bottom, score, class], 
         [left, top, right, bottom, score, class], 
         ...
        ]
        where `left`/`top`/`right`/`bottom` are the bounds of the object, `score` is the
        confidence that the location holds an object, and `label` is the predicted
        object category.
    '''


    scores = F.softmax(classifications, dim=-1).detach().cpu().numpy() #classifications in the range 0-1
    scores = 1 - scores[:, 0] # foreground score for NMS
    
    classifications = classifications.argmax(dim=-1).detach().cpu().numpy()
    regressions = regressions.detach().cpu().numpy()
    detections = []
    for i in range(len(classifications)):
        y, x = i // 16 * 16, (i % 16) * 16
        x_reg, y_reg, w_reg, h_reg = regressions[i]
        x += x_reg * 32
        y += y_reg * 32
        w = np.exp(w_reg) * 32
        h = np.exp(h_reg) * 32
        detections.append((x - w/2, y - h/2, x + w/2, y + h/2, scores[i], classifications[i]))
    return np.array(detections)