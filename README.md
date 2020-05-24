# Object Recognition in Deep Learning
## Introduction

Object classification is a critical task in computer vision applications. It is the task of classifying objects from different object categories. It is useful for Duckiebot to classify the objects in the received images and it can be helpful in tasks such as object detection and tracking. In Duckietown, there are many informative objects such as traffic signs, Duckiebot, duckies, house models, etc. Convolutional Neural Networks (CNN) is one of the variants of neural networks used heavily in the field of Computer Vision. It derives its name from the type of hidden layers it consists of. The hidden layers of a CNN typically consist of convolutional layers, pooling layers, fully connected layers, and normalization layers. It is a great method to use deep learning. Deep Learning is becoming a very popular subset of machine learning due to its high level of performance across many types of data. The Keras library in Python makes it pretty simple to build a CNN.

## Problem Statement

In this project, the classification model of furniture (5 classes) is built using Convolutional Neural Networks (CNN).

## About Dataset

Details | - 
--- | --- 
**Name** | Furniture-images
**Training set** | 4024 images
**Testing set** | 423 images
**Image size** | Varying from (96 x 96) to (450 x 450)
**No. of Classes 5 classes** | chair, sofa, table, swivel-chair, bed
**Dataset** | [Click Here](https://drive.google.com/drive/folders/10Uqr_lQUgamXQs5Z43PKwwuCRx6hG52b?usp=sharing)


## Implementation

The model starts with downloading the dataset from the drive followed by importing related libraries which include TensorFlow Keras, NumPy, matplotlib.pyplot and some more.

## Build the Model
The images that will go into convnet are 150x150 color images on Data Preprocessing, we'll add handling to resize all the images to 150x150 before feeding them into the neural network). We will stack 3 {convolution + relu + maxpooling} modules. Our convolutions operate on 3x3 windows and our maxpooling layers operate on 2x2 windows. Our first convolution extracts 16 filters, the following one extracts 32 filters, and the last one extracts 64 filters. Before the output layer, two fully connected layers are added. The output layer is using softmax activation to classify images.

## Standard HyperParameters


HyperParameters | - 
--- | --- 
**Conv Layers** | 3 Layers
**Filter Size** | 3
**Epochs** | 5
**L2 Regularisation** | 0 (REMOVE)
**Dropout** | 0 (REMOVE)
**Batch** | size 16
**Optimizer** | adam
**Loss** | Categorical crossentropy


## Model Architecture

![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/model_plot.png "Model") 


## Result

Standard results are the results removing all the additional hypermeters (consider standard hyperparameters)  
**Graph between the Accuracies and Epochs**  

![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/standard_results.png "Analysis Graph")  

## Analysis

By changing the hyperparameters of the model, we can see the changes in the results and analyse for the better model. However, changes may have different effects on different datasets and hence it must be chosen carefully. The accuracy tweaks with epochs are shown below as per the changes made. Other hyperparameters in each comparison are standard parameters of the model.  
  
#### Number of convolutional layers  
  
○ 5 layers  
  
![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/convolutional_layer_more.png "Analysis Graph")  
  
○ 2 layers    
  
![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/convolutional_layer_less.png "Analysis Graph")  
  
#### L2 Regularization  
  
○ Add - L2 regularization of 0.01 is added in first 2 hidden layers in the model. 
  
![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/l2_add.png "Analysis Graph")   
○ Remove  
  
![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/standard_results.png "Analysis Graph")  


#### Dropout  
  
○ Add - 0.2 dropout is added between input layer and first hidden layer.  
  
![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/dropout_add.png "Analysis Graph")  
  
○ Remove  
  
![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/standard_results.png "Analysis Graph")  
  
#### Filter / Kernel size  
  
○ Filter size - 6  
  
![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/filter_6.png "Analysis Graph")  
  
○ Filter size - 3  
  
![alt text](https://raw.githubusercontent.com/pragati21p/Object-Recognition-in-Deep-Learning/master/images/standard_results.png "Analysis Graph")  
  
From the above results, we can analyse that adding or removing L2 regularization, changing the layer count or change in filter is not making much changes in the last epoch. Accuracies are in the range [0.83, 0.89]. Although adding dropout of 0.2 is increasing the train accuracy of the model up to 0.945 and validation accuracy up to 0.885 in last epoch.  
  
## References

* [Deep Learning Tutorial | deeplearning.ai](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
* [A friendly introduction to CNN](https://www.youtube.com/watch?v=2-Ol7ZB0MmU)
* [CNN Tutorial | SimpliLearn](https://www.youtube.com/watch?v=Jy9-aGMB_TE)
* [Understanding Neural Networks | Towards Data Science](https://towardsdatascience.com/understanding-neural-networks-from-neuron-to-rnn-cnn-and-deep-learning-cd88e90e0a90)
