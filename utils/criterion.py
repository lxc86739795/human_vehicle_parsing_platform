import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .loss import OhemCrossEntropy2d


class CriterionAll(nn.Module):
    def __init__(self, loss_type='softmax',ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        #self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        if loss_type == 'softmax':
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        if loss_type == 'ohem':
            self.criterion = OhemCrossEntropy2d(ignore_label=ignore_index)
        print (self.criterion)
   
    def parsing_loss(self, preds, target):
        h, w = target.size(1), target.size(2)


        loss = 0
        preds_parsing = preds
        scale_pred = F.interpolate(input=preds_parsing, size=(h, w), mode='bilinear', align_corners=True)
        loss += self.criterion(scale_pred, target)

        return loss

    def forward(self, preds, target):
          
        loss = self.parsing_loss(preds, target) 
        return loss

class CriterionAll_multiloss(nn.Module):
    def __init__(self, loss_type='softmax',ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        #self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.criterion1 = FocalCrossEntropy2d(ignore_index=ignore_index)
        self.criterion2 = OhemCrossEntropy2d(ignore_label=ignore_index)
        print (self.criterion1)
        print (self.criterion2)

   
    def parsing_loss(self, preds, target):
        h, w = target.size(1), target.size(2)


        loss = 0
        preds_parsing = preds
        scale_pred = F.interpolate(input=preds_parsing, size=(h, w), mode='bilinear', align_corners=True)
        loss += self.criterion(scale_pred, target)

        return loss

    def forward(self, preds, target):
          
        loss = self.parsing_loss(preds, target) 
        return loss

class CriterionAll2(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionAll2, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        print (self.criterion)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2
        