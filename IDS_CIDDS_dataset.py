#!/usr/bin/env python
# coding: utf-8

# In[33]:


from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.utils import np_utils 

def build_network():

    models = []
    model = Sequential()
    model.add(Dense(64, input_dim=5))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# In[39]:


from pandas import read_csv
from pandas import DataFrame
import numpy as np

cols_to_use_features = ['Duration', 'Src Pt','Dst Pt','Packets','Flows']
attack_features = read_csv('attacked.csv',usecols=cols_to_use_features, header=0)
attack_features = attack_features.to_numpy()
cols_to_use_labels = ['class']
attack_l = read_csv('attacked.csv',usecols=cols_to_use_labels, header=0)
attack_labels = attack_l.to_numpy()

attack_label_array = []
for i in range(len(attack_labels)):
    if(attack_labels[i] == "normal"):
        attack_label_array.append(0)
    elif (attack_labels[i] == "suspicious"):
        attack_label_array.append(1)
    elif (attack_labels[i] == "unknown"):
        attack_label_array.append(2)


NN = build_network()
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

NN.fit(x=attack_features, y=np_utils.to_categorical(attack_label_array), epochs=100, validation_split=0.1, batch_size=128)



# In[43]:


s = int(len(attack_features) * 0.66)
attack_test_features = attack_features[s:len(attack_features)]

attack_test_labels = attack_label_array[s:len(attack_features)]

preds = NN.predict(attack_test_features)
pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(np_utils.to_categorical(attack_test_labels), axis=1)


# In[45]:


print(pred_lbls)
print(true_lbls)


# In[47]:


from sklearn.metrics import f1_score
f1_score(true_lbls, pred_lbls, average='weighted')


# In[ ]:




