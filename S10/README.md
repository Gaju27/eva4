# Pytorch-cifar 91% Accuracy




# Cifar10  - PyTorch learning rate finder
 > * Aim is to achieve 88% and above accuracy

## Introduction

> Need to achieve 88 and above accuracy with the below requirements.
* Albumentations cutout
* Find Learning rate using (https://github.com/davidtvs/pytorch-lr-finder)
* Implement Pytorch ReduceLROnPlatea

## Prerequisites
* Python 3.7+

* PyTorch 6.0+

## Accuracy

Model         | Accuracy
------------- | -------------
[ResNet18](https://arxiv.org/abs/1512.03385)     | 91 %

## 50 Epochs used
> * Epoch=50 Loss=0.14579269289970398 Batch_id=781 Training Accuracy=97.88:
> * Test set: Average loss: 0.0055, Accuracy: 9096/10000 (91%) 

## Before start we need to install below packages
* pip install pytorch-gradcam
* pip install albumentation
* pip install torch-lr-finder

## Graphs

### Loss Graph

![loss_graph](https://github.com/Gaju27/eva4/blob/master/S10/loss_graph.png)

### Learning rate Graph

![learningrate](https://github.com/Gaju27/eva4/blob/master/S10/learning%20rate_graph.png)

### Misclassified Images

![Misclassified](https://github.com/Gaju27/eva4/blob/master/S10/predictions/incorrect_predictions.png)

### GradCam
![gradcam](https://github.com/Gaju27/eva4/blob/master/S10/images/GradCam.JPG)
