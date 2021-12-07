#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json


# In[6]:


import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.special import softmax

class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3072, 1000)
    # TODO
    # You need to add the second layer's parameters
        self.fc2 = torch.nn.Linear(1000, 10)
    def forward(self, X):
        batch_size = X.size(0)
    # This next line reshapes the tensor to be size (B x 3072)
    # so it can be passed through a linear layer.
        X = X.view(batch_size, -1)
    # TODO
    # You need to pass X through the two linear layers and relu
    # then return the final scores
    
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        return X


# In[5]:


class Convolutional(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3,
                                 out_channels=7,
                                 kernel_size=3,
                                 stride=1,
                                 padding=0)

    # You need to add the pooling, second convolution, and
    # three linear modules here
    self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2)
    self.conv2 = torch.nn.Conv2d(in_channels=7,
                                 out_channels=16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=0)
    self.fc1 = torch.nn.Linear(2704, 130)
    self.fc2 = torch.nn.Linear(130, 72)
    self.fc3 = torch.nn.Linear(72, 10)

  def forward(self, X):
    batch_size = X.size(0)
    X = self.conv1(X)
    X = self.pool(X)
    X = self.conv2(X)
    X = F.relu(X)
    X = X.view(X.size(0), -1)
    X = self.fc1(X)
    X = F.relu(X)
    X = self.fc2(X)
    X = F.relu(X)
    X = self.fc3(X)
    X = torch.sigmoid(X)

    return X


# In[ ]:


from collections import defaultdict
def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    vocab = set()
    vocab_freq = {}
    vocab.add("<unk>")
    # check frequencies
    for doc in D:
      for token in doc:
        if token in vocab_freq.keys():
          vocab_freq[token] += 1
        else:
          vocab_freq[token] = 1
    for token, freq in vocab_freq.items():
      if vocab_freq[token] > 1:
        vocab.add(token)
          
    return vocab
# In[1]:


class BBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the binary bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        feat_dict = {}
        for token in doc:
            if token in vocab:
                feat_dict[token] = 1
            else:
                feat_dict["<unk>"] = 1
        return feat_dict


# In[2]:


class CBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the count bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        feat_dict = {}
        for token in doc:
            if token in vocab:
                if token in feat_dict:
                    feat_dict[token] += 1
                else:
                    feat_dict[token] = 1
            else:
                if "<unk>" in feat_dict:
                    feat_dict["<unk>"] += 1
                else:
                    feat_dict["<unk>"] = 1
        return feat_dict


# In[3]:


def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    list_size = len(D)
    idf_scores = {}
        
    for doc in D:
        for token in vocab:
            if token in doc:
                if token not in idf_scores:
                    idf_scores[token] = 1
                else:
                    idf_scores[token] = idf_scores[token] + 1
    
        
    # account for <unk>
    idf_scores["<unk>"] = 0
    for doc in D:
        for token in doc:
            if token not in vocab:
                idf_scores["<unk>"] = idf_scores["<unk>"] + 1
                break
                
    for token, val in idf_scores.items():
        idf_scores[token] = numpy.log((list_size/val))
        
    return idf_scores
    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODOß
        feat_dict_tf_idf = {}
        cbow = CBoWFeaturizer()
        feat_dict_tf =cbow.convert_document_to_feature_dictionary(doc, vocab)
        for token, tf in feat_dict_tf.items():
          feat_dict_tf_idf[token] = feat_dict_tf[token] * self.idf[token]
        return feat_dict_tf_idf

# In[4]:



def load_dataset(file_path):
    D = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            D.append(instance['document'])
            y.append(instance['label'])
    return D, y

def convert_to_features(D, featurizer, vocab):
    X = []
    for doc in D:
        X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
    return X


# In[8]:


from collections import defaultdict
def train_naive_bayes(X, y, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    vocab is the set of vocabulary tokens.
    
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the innner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    y_freq = {}
    p_y = {}
    p_v_y = {}
    total_labels = len(y)
    vocab_size = len(vocab)
    for label in y:
      p_v_y[label] = {}
      if label not in y_freq.keys():
        y_freq[label] = 1
      else:
        y_freq[label] = y_freq[label] + 1
    for label, freq in y_freq.items():
      p_y[label] = y_freq[label] / total_labels
    
    #calculate the denominator
    f_d_w = {}
    for label in y:
        f_d_w[label] = k * vocab_size
        for v in vocab:
            for i in range(len(X)):
                if y[i] == label:
                    if v in X[i].keys(): # check if current word in vocab is in current doc
                        f_d_w[label] = f_d_w[label] + X[i][v]

    #calculate the numerator
    for label in y: #all y we consider for the P(v|y)'s
        #f_d_v = {}
        #f_d_v[label] = {} # numerator, dictionary for all tokens in the feature space
      # f_d_w = 0 # denominator
        for v in vocab:
            p_v_y[label][v] = k
            for i in range(len(X)):
                if y[i] == label: # consider only all documents with current label of interest
                #for token, val in X[i].items(): #X[i] represents a document
                    if v in X[i].keys(): # for a given token of interrest
                        p_v_y[label][v] = p_v_y[label][v] + X[i][v]
                        
      # p_v_y_inner = {}
      #find the P(v|y)'s plus Laplace smoothing
    for label, val in p_v_y.items():
        for token, y in val.items():
            p_v_y[label][token] = y / (f_d_w[label])
      # p_v_y[label] = p_v_y_inner

    return p_y, p_v_y

# In[ ]:


def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    # TODO
    predictions = []
    confidences = []
    
    for doc in D:
      argmax_pyd = 0
      prediction = 0
      map = {}
      arr = []
      labels = []
      for label, val in p_y.items():
        prob_y = numpy.log(val)
        p_y_d = prob_y
        pyd_normal = val
        for token in doc:
        # check for "<unk>"
          if label in p_v_y:
            if token not in p_v_y[label]:
                token = "<unk>"
          p_y_d = p_y_d + numpy.log(p_v_y[label][token])
          # pyd_normal = pyd_normal * p_v_y[label][token]
        labels.append(p_y_d)
        map[label] = p_y_d
        arr.append(float(p_y_d))
      
        #if pyd_normal > argmax_pyd: #prediction is the label corresponding to max prob
          #argmax_pyd = p_y_d
          #prediction = label
      
      predictions.append(np.argmax(labels))
      
      confidences.append(max(softmax(arr)))
      
    
    return predictions, confidences
    


