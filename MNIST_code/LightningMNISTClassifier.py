# Neural network for MNIST digit classification (works for the dataset
# contains only 2,3,6)
## Standard libraries
import os
import json
import math
import numpy as np
import random
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Lightning
import pytorch_lightning as pl


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, lr_rate=1e-4):
        super(LightningMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 3)
        self.lr_rate = lr_rate

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        logs = {'train_loss': loss}
        self.log('train_loss', loss)
        return {'loss': loss, 'log': logs}

    def validation_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        
        pred = logits.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum()/y.shape[0]
        
        self.log('val_loss', loss)
        self.log('Accuracy',correct)
        return {'val_loss': loss, 'Accuracy':correct}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99),
                    'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]