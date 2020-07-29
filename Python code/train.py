import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from Loss_acc_rec_pres import softmax_focal_loss
from boxcomparing import generate_targets
from Model import forward_data
from Loss_acc_rec_pres import compute_ap
from Loss_acc_rec_pres import compute_ar



def train_epoch(train_data, train_boxes, train_labels, anchor_boxes, model, optim,
                val_data, val_boxes, val_labels):


    torch.set_grad_enabled(True)
    idxs = np.arange(len(train_data))
    np.random.shuffle(idxs)
    
    for idx in idxs:
        img, boxes, labels = train_data[idx], train_boxes[idx], train_labels[idx]
        cls_targs, reg_targs = generate_targets(anchor_boxes, boxes, labels)
        classifications, regressions = forward_data(img[np.newaxis])

        # ignore the (ambiguous) anchors
        mask = np.where(cls_targs > -1)[0]
        cls_loss = softmax_focal_loss(classifications[mask].cpu(), cls_targs[mask], 
                                      alpha=0.25, gamma=2)
        
       # only give a regression loss for positive boxes
        mask = np.where(cls_targs > 0)[0]
        reg_loss = F.smooth_l1_loss(regressions[mask].cpu(), reg_targs[mask])
        
        total_loss = cls_loss + reg_loss
        total_loss.backward()
        optim.step()
        optim.zero_grad()

        ap = compute_ap(classifications, regressions, boxes, labels[:, np.newaxis])
        ar = compute_ar(classifications, regressions, boxes, labels[:, np.newaxis])
        
        #only for jupyter when u have noggin plot initied
        #plotter.set_train_batch({'loss': total_loss.item(),
        #                         'reg_loss': reg_loss.item(),
         #                        'cls_loss': cls_loss.item(),
          #                       'Precision': ap, 
           #                      'Recall': ar}, 1)
        
    torch.set_grad_enabled(False)
    for idx in range(len(val_data)):
        img, boxes, labels = val_data[idx], val_boxes[idx], val_labels[idx]
        cls_targs, reg_targs = generate_targets(anchor_boxes, boxes, labels)
        classifications, regressions = forward_data(img[np.newaxis])
        
        mask = np.where(cls_targs > -1)[0]
        cls_loss = softmax_focal_loss(classifications[mask].cpu(), cls_targs[mask], 
                                      alpha=0.25, gamma=2)
        mask = np.where(cls_targs > 0)[0]
        reg_loss = F.smooth_l1_loss(regressions[mask].cpu(), reg_targs[mask])
        total_loss = cls_loss + reg_loss
        
        ap = compute_ap(classifications, regressions, boxes, labels[:, np.newaxis])
        ar = compute_ar(classifications, regressions, boxes, labels[:, np.newaxis])
        #plotter.set_test_batch({'loss': total_loss.item(),
         #                        'reg_loss': reg_loss.item(),
          #                       'cls_loss': cls_loss.item(),
           #                      'Precision': ap,
            #                     'Recall': ar}, 1)
        
    #plotter.set_train_epoch()
   # plotter.set_test_epoch()