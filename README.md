# Multi-Class Image Classification and Naive Bayes Document Classification

## Image Classification
### 
Implements two different neural network architectures to do image classification with the CIFAR-10 image dataset. Goal is to better understand how to build different neural neutwork architectures for image data and implement a multi-class classifier. Training ussing GPU in Google Colab. 

###
The CIFAR-10 dataset contains tens of thousands of images that have been classified in ten different categories, such as “airplane,” “cat”, and “horse.” Each image is 32 × 32 pixels large and is represented uses the RGB format. Therefore, each image is represented as a (3 × 32 × 32) matrix where each value is a number between 0 and 255 (inclusive). Each of the three channels corresponds to the red, green, or blue channels of the image.

###
Trains two different neural networks with stochastic gradient descent with several different learning rates to find the best one. I use the torch.nn.CrossEntropyLoss loss function and train for 200 epochs. For each network, I compare each of the learning rates by plotting the following data:
1. The average loss per training instance on each epoch3
2. The average loss per validation instance on each epoch 
3. The accuracy on the validation dataset on each epoch

See experiment graphs in folder.

## Naive Bayes
