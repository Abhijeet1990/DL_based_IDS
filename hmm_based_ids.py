#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from hmmlearn import hmm
import pandas as pd
from time import time
import random


# In[33]:


# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=6, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        #self.algorithm = algorithm
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        #print(self.model.transmat_.shape)
        return self.model.score(input_data)
    



# In[34]:



col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

dataset = pd.read_csv("kddcup.data_10_percent_corrected", names = col_names)

num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]


# Create a list of HMM models
hmm_models = []

X = np.array([])
y_words = []

# for each type of attacks and normal traffic type
for row_by_label in dataset.label.unique():
    
    # get the traffic from the dataset for that specific attack label
    dataset_per_traffic_type = dataset[dataset.label == row_by_label]
    random_pick = 100

    # randomly pick 100 sample for the specific attack type
    try:
        sample_rows = random.sample(list(dataset_per_traffic_type.index), random_pick)
    except:
        random_pick = len(dataset_per_traffic_type)
        sample_rows = random.sample(list(dataset_per_traffic_type.index), random_pick)

    # get the data for those specific rows
    data_sample_per_traffic_type = dataset_per_traffic_type.ix[sample_rows]

    #print(data_sample_per_traffic_type)
    # get the required features from those data sample for that specific traffic type
    feature_per_traffic_type = data_sample_per_traffic_type[num_features]
    
    if len(X) == 0:
        X = feature_per_traffic_type
    else:
        X = np.append(X, feature_per_traffic_type, axis=0)

    print(X)
    y_words.append(row_by_label)
    hmm_trainer = HMMTrainer()

    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, row_by_label))
    hmm_trainer = None
    
result = {'correct' : 0, 'incorrect' : 0}
result_label = []


for row_by_label in dataset.label.unique():
    try:
        dataset_per_traffic_type = dataset[dataset.label == row_by_label]
        
        # randomly pick one sample from the dataset of that attack type
        random_pick = 1
        
        data_test_row = random.sample(list(dataset_per_traffic_type.index), random_pick)
        data_test = dataset_per_traffic_type.ix[data_test_row]
        data_test_feature = data_test[num_features].astype(float)
        
        print(data_test_feature)
    
        # Define variables
        max_score = -10000
        output_label = None

        # Iterate through all HMM models and pick 
        # the one with the highest score
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(data_test_feature)
            print(score)
            if score > max_score:
                max_score = score
                output_label = label

        label_res = {'origin' : row_by_label, 'predicted' : output_label}
        print(label_res)
        result_label.append(label_res)
        
            
    except Exception as e:
        raise e

print("============================================================================================")
print("result")
for each_ in result_label:
    #print("Origin : {} | Predicted : {}".format(each_['origin'], each_['predicted']))
    print(each_)
    if each_['origin'] == each_['predicted']:
        result['correct'] += 1
        print('correct')
    else:
        result['incorrect'] += 1
        print('incorrect')
            
    print('')
    
print("result : {} correct {} data incorrect".format(result['correct'], result['correct'] + result['incorrect']))
correct_precentage = float(result['correct']) / float(result['correct'] + result['incorrect']) * 100
print(str(correct_precentage) + '%')





