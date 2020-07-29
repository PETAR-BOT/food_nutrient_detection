from boxcomparing import compute_iou
import numpy as np


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

