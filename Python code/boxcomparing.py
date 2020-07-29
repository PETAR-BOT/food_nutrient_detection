import numpy as np
from matplotlib.patches import Rectangle


def compute_iou(boxes, truth):
    ''' Return the intersection over union between two arrays of boxes.

    Parameters
    ----------
    boxes : numpy.ndarray, shape=(N, 4)
        The predicted boxes, in xyxy format.

    truth : numpy.ndarray, shape=(K, 4)
        The ground-truth boxes, in xyxy format.

    Returns
    -------
    numpy.ndarray, shape=(N, K)
        The overlap between the predicted and ground-truth boxes
    '''

    N = boxes.shape[0] #amount of predicted boxes
    K = truth.shape[0] #amount of actual boxes 
    ious = np.zeros((N, K), dtype= np.float32)

    for k in range(K):
        truth_area = (truth[k, 2] - truth[k, 0]) * (truth[k, 3] - truth[k, 1])
        for n in range(N):
            width_overlap = min(boxes[n, 2], truth[k, 2]) - \
                            max(boxes[n, 0], truth[k, 0])
            if width_overlap > 0:
                height_overlap = min(boxes[n, 3], truth[k, 3]) - \
                                 max(boxes[n, 1], truth[k, 1])
                if height_overlap > 0:
                    union = (boxes[n, 2] - boxes[n, 0]) * \
                            (boxes[n, 3] - boxes[n, 1]) + truth_area - \
                            (width_overlap * height_overlap)
                    ious[n, k] = width_overlap * height_overlap / union
    return ious

def generate_targets(anchor_boxes, truth_boxes, labels):
    ''' Determine the correct label and regression target for each anchor box given
    a set of truth boxes.
    
    Parameters
    ----------
    anchor_boxes : numpy.ndarray, shape=(N, 4)
        The anchor boxes, in xyxy format.
        
    truth_boxes : numpy.ndarray, shape=(K, 4)
        The ground-truth boxes, in xyxy format.
        
    labels : numpy.ndarray, shape=(K,)
        The correct object label for each of the ground-truth boxes.
        
    Returns
    -------
    Tuple[numpy.ndarray shape=(N,), numpy.ndarray shape=(N, 4)]
        (classification, regressions) for each anchor box in `anchor_boxes`.
    '''
    ious = compute_iou(anchor_boxes, truth_boxes) # NxK
    max_ious = ious.max(axis=1)                   # N IoUs
    max_idxs = ious.argmax(axis=1)                # N indices
    
    target_boxes = truth_boxes[max_idxs]

    # we want to regress the xy offset of the center and the wh offsets
    target_centers = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
    anchor_centers = (anchor_boxes[:, :2] + anchor_boxes[:, 2:]) / 2
    target_wh = target_boxes[:, 2:] - target_boxes[:, :2]
    anchor_wh = anchor_boxes[:, 2:] - anchor_boxes[:, :2]
    
    xy = (target_centers - anchor_centers) / anchor_wh # predict wrt anchor box wh
    wh = np.log(target_wh / anchor_wh)                 # predicting log keeps values small
    
    targets_reg = np.hstack([xy, wh])
    targets_cls = labels[max_idxs]
    targets_cls[max_ious < 0.3] = -1 # if our anchor has medium overlap, ignore - ambiguous
    targets_cls[max_ious < 0.2] = 0  # if our anchor has little overlap, negative

    return targets_cls, targets_reg

def add_detection(ax, box, label):
    ''' Add a detection box to the provided axes object.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        The set of axes on which to add the detection box.
        
    box : Iterable, shape=(4,)
        The bounds of the box box, in (left, top, right, bottom) format.

    label : Union[int, str]
        The label to apply to the box.
    '''
    x, y, x2, y2 = box
    w = x2 - x
    h = y2 - y
    ax.add_patch(Rectangle((x, y), w, h, color='r', lw=2, fill=None))
    try:
        label = int(label)
        label = 'Bread' if label == 1 else label
        label = 'Dairy product' if label == 2 else label
        label = 'Dessert' if label == 3 else label
        label = 'Egg' if label == 4 else label
        label = "Fried food" if label == 5 else label
        label = 'Meat' if label == 6 else label 
        label = 'Noodles/Pasta' if label == 7 else label
        label = 'Rice' if label == 8 else label
        label = 'Seafood' if label == 9 else label
        label = 'Soup' if label == 10 else label
        label = 'Vegetable/Fruit' if label == 11 else label
        
    except:
        label = str(label)
    ax.annotate(label, (x, y), color='r')


def non_max_suppression(detections, threshold=0.7):
    ''' Apply non-maximum suppression to the detections provided with a given threshold.
    
    Parameters
    ----------
    detections : np.ndarray[Real], shape=(N, 5)
        The detection boxes to which to apply NMS, in (left, top, right, bottom, score) 
        format.
        
    threshold : float ∈ [0, 1], optional (default=0.7)
        The IoU threshold to use for NMS.
        
    Returns
    -------
    numpy.ndarray[int], shape=(k,)
        The indices of `detections` to keep.
    '''
    x1s = detections[:, 0] # left
    y1s = detections[:, 1] # top
    x2s = detections[:, 2] # right
    y2s = detections[:, 3] # bottom

    areas = (x2s - x1s) * (y2s - y1s)
    order = detections[:, 4].argsort()[::-1] # highest to lowest score

    keep = [] # which detections are we going to keep?
    while order.size > 0:
        i = order[0]
        keep.append(i)
        all_others = order[1:] # everything except the current box
        width_overlaps = np.maximum(0, np.minimum(x2s[i], x2s[all_others])  -
                                       np.maximum(x1s[i], x1s[all_others]))
        height_overlaps = np.maximum(0, np.minimum(y2s[i], y2s[all_others]) - 
                                        np.maximum(y1s[i], y1s[all_others]))
        intersections = width_overlaps * height_overlaps
        ious = intersections / (areas[i] + areas[all_others] - intersections)

        # +1 to counteract the offset all_others = order[1:]
        order = order[np.where(ious <= threshold)[0] + 1]

    return keep