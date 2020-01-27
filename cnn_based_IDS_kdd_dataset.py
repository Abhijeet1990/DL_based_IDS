#!/usr/bin/env python
# coding: utf-8


# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# For the original '99 KDD dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html


# In[3]:


with open('kddcup.names', 'r') as infile:
    kdd_names = infile.readlines()
kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]


# In[4]:


# The Train+/Test+ datasets include sample difficulty rating and the attack class


# In[5]:


kdd_cols += ['class', 'difficulty']


# In[6]:


kdd = pd.read_csv('KDDTrain+.txt', names=kdd_cols)
kdd_t = pd.read_csv('KDDTest+.txt', names=kdd_cols)


kdd_cols = [kdd.columns[0]] + sorted(list(set(kdd.protocol_type.values))) + sorted(list(set(kdd.service.values))) + sorted(list(set(kdd.flag.values))) + kdd.columns[4:].tolist()


# In[10]:


attack_map = [x.strip().split(',') for x in open('Attack Types.csv', 'r')]
attack_map = {k:v for (k,v) in attack_map}



# In[13]:


kdd['class'] = kdd['class'].replace(attack_map)
kdd_t['class'] = kdd_t['class'].replace(attack_map)


# In[14]:


def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)


# In[15]:


def log_trns(df, col):
    return df[col].apply(np.log1p)


# In[16]:


cat_lst = ['protocol_type', 'service', 'flag']
for col in cat_lst:
    kdd = cat_encode(kdd, col)
    kdd_t = cat_encode(kdd_t, col)


# In[17]:


log_lst = ['duration', 'src_bytes', 'dst_bytes']
for col in log_lst:
    kdd[col] = log_trns(kdd, col)
    kdd_t[col] = log_trns(kdd_t, col)


# In[18]:


kdd = kdd[kdd_cols]
for col in kdd_cols:
    if col not in kdd_t.columns:
        kdd_t[col] = 0
kdd_t = kdd_t[kdd_cols]


# In[19]:


# Now we have used one-hot encoding and log scaling


# In[20]:


kdd.head()


# In[21]:


difficulty = kdd.pop('difficulty')
target = kdd.pop('class')
y_diff = kdd_t.pop('difficulty')
y_test = kdd_t.pop('class')


# In[22]:


y_test.head(200)


# In[23]:


target = pd.get_dummies(target)
y_test = pd.get_dummies(y_test)


target = target.values
train = kdd.values
test = kdd_t.values
y_test = y_test.values


# In[27]:


# We rescale features to [0, 1]


# In[28]:


min_max_scaler = MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)


for idx, col in enumerate(list(kdd.columns)):
    print(idx, col)


# ## The Model

# In[32]:


from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout,Conv1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding


# In[34]:


def build_network():

    models = []
    model = Sequential()
    model.add(Dense(64, input_dim=122))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_cnn_network():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, strides=1,
                 activation='relu',
                 input_shape=(122,1)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[35]:


# We use early stopping on a holdout validation set


# In[46]:


NN = build_network()
#NN = build_cnn_network()
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')


# In[50]:


#NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=128, callbacks=[early_stopping])
#NN.fit(x=np.expand_dims(train,axis=2), y=target, epochs=100, validation_split=0.1, batch_size=128)
NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=128)


# ## The Performance

# In[53]:


print(y_test)
from sklearn.metrics import confusion_matrix
preds = NN.predict(test)
#preds = NN.predict(np.expand_dims(test,axis=2))
pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test, axis=1)


# In[56]:


print(test.shape)
print(y_test.shape)
#NN.evaluate(np.expand_dims(test,axis=2), y_test)
NN.evaluate(test, y_test)


# In[57]:


# With the confusion matrix, we can aggregate model predictions
# This helps to understand the mistakes and refine the model


# In[58]:


confusion_matrix(true_lbls, pred_lbls)


# In[59]:


from sklearn.metrics import f1_score
f1_score(true_lbls, pred_lbls, average='weighted')


# In[60]:


from sklearn.metrics import multilabel_confusion_matrix
conf = multilabel_confusion_matrix(true_lbls, pred_lbls)

for i in range(conf.shape[0]):
    tn = conf[i][0][0]
    fp = conf[i][0][1]
    fn = conf[i][1][0]
    tp = conf[i][1][1]
    acc = (tp+tn)/(tp+tn+fp+fn)
    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    print("Accuracy",float("{0:5f}".format(acc))*100)


# In[ ]:




