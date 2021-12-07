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

###
The dataset used for the experiments is be a sentiment analysis dataset from IMDb. Each document is a review of a movie and has been labeled as either a positive or negative review. 

Representations We will use fd(v) to denote the value for feature v and document d. The three ways that you should represent each document (and the corresponding Python
functions to implement) are described below.
Note that we have already implemented the first two, (1) Binary Bag-of-Words and (2)
Count Bag-of-Words for you. It is up to you to implement TF-IDF.
1. Binary Bag-of-Words Each document should be represented with binary features, one for each token in the vocabulary.
􏰄1 ifv∈d
fd(v) = 0 otherwise (1)
Since you are using a sparse representation, you do not need to remember which tokens are not present in each document, only those which are present.
2. Count Bag-of-Words Instead of having a binary feature for each token, you should keep count of how many times the token appears in the document, a quantity known as term frequency and denoted tf(d,v).
fd(v) = tf(d,v) (2)
3. TF-IDF Model The final representation will use the TF-IDF score of each token. The TF-IDF score combines how frequently the word appears in the document as well as how infrequently that word appears in the document collection as a whole. First, the inverse document frequency (IDF) of a token is defined as
 2.3
idf(v) = log |D| (3) |{d : v ∈ d, d ∈ D}|
Here, D is the set of documents6 and the denominator is the number of documents that the token v appears in. Use the numpy.log() function to compute the log. Then, the TF-IDF feature you should use is
fd(v) = tf(d, v) × idf(v) (4
