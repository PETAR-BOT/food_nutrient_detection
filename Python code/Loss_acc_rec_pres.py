from boxcomparing import compute_iou
import numpy as np
from Model import compute_detections
from boxcomparing import non_max_suppression

def compute_recall(prediction_detections, truth_detections, threshold=0.5):
    ''' Compute the recall of predictions given targets.

    remember that recall is true pos predicted : the acutal # of tru pos

    Parameters
    ----------
    prediction_detections : numpy.ndarray, shape=(N, 5)
        The predicted objects, in (left, top, right, bottom, class) format.
        
    truth_detections : numpy.ndarray, shape=(K, 5)
        The ground-truth objects in (left, top, right, bottom, class) format.
        
    threshold : Real, optional (default=0.5)
        The IoU threshold at which to compute average precision.
        
    Returns
    -------
    float
        The average recall (AR) for the given detections and truth.
    '''
    ious = compute_iou(prediction_detections[:, :4], truth_detections[:, :4])
    max_ious = ious.max(axis=1) # N IoUs
    max_idxs = ious.argmax(axis=1)
    
    predictions = prediction_detections[:, -1]
    truths = truth_detections[:, -1]
    target_labels = truths[max_idxs]
    correct = predictions == target_labels
    true_pos = correct[np.logical_and(max_ious >= threshold, target_labels > 0)].sum()
    
    wrong = predictions != target_labels
    false_negatives = wrong[np.logical_and(max_ious >= threshold, predictions == 0)].sum()
    false_negatives += (ious.max(axis=0) < threshold).sum()
    if true_pos + false_negatives == 0:
        return 1
    return true_pos / (true_pos + false_negatives)

def compute_precision(prediction_detections, truth_detections, threshold=0.5):
    ''' Compute the precision of predictions given targets.
    
    Remember that precision is the true pos : # predicted pos

    Parameters
    ----------
    prediction_detections : numpy.ndarray, shape=(N, 5)
        The predicted objects, in (left, top, right, bottom, class) format.
        
    truth_detections : numpy.ndarray, shape=(K, 5)
        The ground-truth objects in (left, top, right, bottom, class) format.
        
    threshold : Real, optional (default=0.5)
        The IoU threshold at which to compute average precision.
        
    Returns
    -------
    float
        The average precition (AP) for the given detections and truth.
    '''
    ious = compute_iou(prediction_detections[:, :4], truth_detections[:, :4])
    max_ious = ious.max(axis=1)
    max_idxs = ious.argmax(axis=1)
    
    predictions = prediction_detections[:, -1]
    truths = truth_detections[:, -1]
    target_labels = truths[max_idxs]

    correct = predictions == target_labels
    true_pos = correct[np.logical_and(max_ious >= threshold, target_labels > 0)].sum()
    total_preds = (predictions > 0).sum()
    return true_pos / total_preds if total_preds > 0 else 0

def compute_ar(classifications, regressions, boxes, labels, nms_threshold=0.3, ar_threshold=0.5):
    ''' Compute the average recall of the provided detections.
    
    Parameters
    ----------
    classifications : mygrad.Tensor, shape=(N, 12)
        The classification predictions at each feature map location in the image.
        
    regressions : mygrad.Tensor, shape=(N, 12)
        The regression predictions at each feature map location in the image.
        
    boxes : numpy.ndarray, shape=(K, 12)
        The bounding boxes of the objects in the image, in (left, top, right, bottom) format.
        
    labels : numpy.ndarray, shape=(K, 1)
        The labels of the objects in the image.
        
    nms_threshold : Real, optional (default=0.3)
        The non-maximum suppression threshold to apply to the objects before computing ar.
        
    ar_threshold : Real, optional (default=0.5)
        The threshold at which to count a predicted box as overlapping a ground-truth object.
        
    Returns
    -------
    float
        The average precision for the given image.
    '''


    detections = compute_detections(classifications, regressions)
    detections = detections[non_max_suppression(detections[:, :5], nms_threshold)]
    detections = np.hstack((detections[:, :4], detections[:, -1:])) # remove score; keep label
    
    truth_detections = np.hstack((boxes, labels))
    if len(detections) > 0:
        return compute_recall(detections, truth_detections, ar_threshold)
    return 0

def compute_ap(classifications, regressions, boxes, labels, nms_threshold=0.3, ap_threshold=0.5):
    ''' Compute the average precision of the provided detections.
    
    Parameters
    ----------
    classifications : mygrad.Tensor, shape=(N, 12)
        The classification predictions at each feature map location in the image.
        
    regressions : mygrad.Tensor, shape=(N, 12)
        The regression predictions at each feature map location in the image.
        
    boxes : numpy.ndarray, shape=(K, 12)
        The bounding boxes of the objects in the image, in (left, top, right, bottom) format.
        
    labels : numpy.ndarray, shape=(K, 1)
        The labels of the objects in the image.
        
    nms_threshold : Real, optional (default=0.3)
        The non-maximum suppression threshold to apply to the objects before computing AP.
        
    ap_threshold : Real, optional (default=0.5)
        The threshold at which to count a predicted box as overlapping a ground-truth object.
        
    Returns
    -------
    float
        The average precision for the given image.
    '''
    
    detections = compute_detections(classifications, regressions)
    detections = detections[non_max_suppression(detections[:, :5], nms_threshold)]
    detections = np.hstack((detections[:, :4], detections[:, -1:])) # remove score; keep label
    
    truth_detections = np.hstack((boxes, labels))
    if len(detections) > 0:
        return compute_precision(detections, truth_detections, ap_threshold)
    return 0