#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
from keras.layers import Dropout,LSTM,Dense
from keras.models import Sequential
from keras.callbacks import History,ModelCheckpoint,EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix, f1_score, precision_score, recall_score,classification_report,roc_curve
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.optimizers import Adam
import pandas as pd


# In[2]:


class lstm_class:
    def __init__(self, alpha, batch_size, cell_size, dropout, sequence_length):
        self.alpha = alpha # learning rate
        self.batch_size = batch_size # no. of batches to use for training/validation or testing
        self.cell_size = cell_size # size of cell state
        # At each training stage, individual nodes are either dropped out of the net with probability 1-p or kept with probability p, so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed.
        self.dropout= dropout # dropout rate.. dropout refers to ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random. By “ignoring”, I mean these units are not considered during a particular forward or backward pass. 
        self.sequence_length = sequence_length # Number of features in the dataset
        
    def create_model(self):
        model = Sequential()
        model.add(LSTM(2, input_shape=(1, self.sequence_length), return_sequences=True, activation='sigmoid'))
        #model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.cell_size, inner_activation='hard_sigmoid', activation='sigmoid'))
        #model.add(Dropout(self.dropout))
        model.add(Dense(activation='sigmoid', units=5))
        print('Compiling...')
        adam=Adam(lr=self.alpha)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model
    
    def train(self,checkpoint_path,model, epochs, X_train, y_train,model_path,batch_size,X_val,y_val,result_path):
        
        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)
        # checkpoint


        filepath = os.path.join(checkpoint_path, '{}-weights.-{}.hdf5'.format(self.cell_size,self.sequence_length))
        #TODO ADD ARGUMENT TO LOAD WEIGHTS
        #filepath="weights.best.hdf5"
        #set the callback variables
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=10)
        history = History()
        callbacks_list = [checkpoint, history,early_stopping_monitor]
        print(model.summary())
        print('Fitting model...')
        #TODO KFOLD VALIDATION
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val,y_val),callbacks=callbacks_list)
        #Evaluation using the trained data
        score, acc = model.evaluate(X_train, y_train, batch_size=batch_size)
        #predict classes of the trained data
        y_pred = model.predict_classes(X_train, batch_size=batch_size)
        print("test data , score ,accu")
        print('Test score:', score)
        print('Test accuracy:', acc)
        print("train data, score ,accu")
        #saving the model
        file = os.path.join(model_path, '{}-trained-model-{}.h5'.format(self.alpha, float('%.2f' % acc)))
        model.save(file)
        #saving the labels to npy files under result_path
        lstm_class.save_labels(predictions=y_pred,actual=y_train,result_path=result_path,phase='training',acc=acc)

        # summarize history for training accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    @staticmethod
    def predict(batch_size, X_test, y_test,result_path,model_path):
        termwidth, fillchar = 124, '-'
        print("Loading Model...")
        model=load_model(model_path)
        #print(" Evaluating ".center(termwidth,fillchar))
        #score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
        #print('\n')
        print(" Predicting ".center(termwidth,fillchar))
        y_pred = model.predict_classes(X_test, batch_size=batch_size)
        y_test_mod = (y_test!=0).argmax(axis=1)
        # classification report
        #print("\nClassification Report")
        conf = multilabel_confusion_matrix(y_test_mod,y_pred)
        print(conf)
        for i in range(conf.shape[0]):
            tn = conf[i][0][0]
            fp = conf[i][0][1]
            fn = conf[i][1][0]
            tp = conf[i][1][1]
            acc = (tp+tn)/(tp+tn+fp+fn)
            fpr = fp/(fp+tn)
            tpr = tp/(tp+fn)
            print(fpr)
            print(tpr)
            print("Accuracy",float("{0:5f}".format(acc))*100)
        #print(classification_report(y_pred=y_pred, y_true=y_test))
        #print(conf)
        print("Saving results to: " + result_path)
        lstm_class.save_labels(predictions=y_pred,actual=y_test,result_path=result_path,phase='testing',acc=acc)
        #file = os.path.join('F:\RNN-LSTM-Network-Intrusion\modelSaves', '{}-tested model-{}.h5'.format(batch_size, float('%.2f' % acc)))
        #model.save(file)

    @staticmethod
    def save_labels(predictions, actual, result_path, phase,acc):
        print(predictions.shape)
        print(actual.shape)      
        print(predictions[0:5])    
        actual_mod = (actual!=0).argmax(axis=1)
        print(actual_mod[0:5])
        # Concatenate the predicted and actual labels
        #labels = np.concatenate((predictions, actual_mod), axis=1)
        labels = np.column_stack((predictions, actual_mod))
        
        if not os.path.exists(path=result_path):
            os.mkdir(path=result_path)

        # save every labels array to NPY file
        np.save(file=os.path.join(result_path, '{}-LSTM-Results-{}.npy'.format(phase, float('%.2f'%acc))), arr=labels)


# In[3]:



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

with open('kddcup.names', 'r') as infile:
    kdd_names = infile.readlines()
kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]

kdd_cols += ['class', 'difficulty']

kdd = pd.read_csv('KDDTrain+.txt', names=kdd_cols)
kdd_t = pd.read_csv('KDDTest+.txt', names=kdd_cols)

kdd_cols = [kdd.columns[0]] + sorted(list(set(kdd.protocol_type.values))) + sorted(list(set(kdd.service.values))) + sorted(list(set(kdd.flag.values))) + kdd.columns[4:].tolist()

attack_map = [x.strip().split(',') for x in open('Attack Types.csv', 'r')]
attack_map = {k:v for (k,v) in attack_map}

kdd['class'] = kdd['class'].replace(attack_map)
kdd_t['class'] = kdd_t['class'].replace(attack_map)

def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)

def log_trns(df, col):
    return df[col].apply(np.log1p)

cat_lst = ['protocol_type', 'service', 'flag']
for col in cat_lst:
    kdd = cat_encode(kdd, col)
    kdd_t = cat_encode(kdd_t, col)
    
log_lst = ['duration', 'src_bytes', 'dst_bytes']
for col in log_lst:
    kdd[col] = log_trns(kdd, col)
    kdd_t[col] = log_trns(kdd_t, col)
    
kdd = kdd[kdd_cols]
for col in kdd_cols:
    if col not in kdd_t.columns:
        kdd_t[col] = 0
kdd_t = kdd_t[kdd_cols]

difficulty = kdd.pop('difficulty')
target = kdd.pop('class')
y_diff = kdd_t.pop('difficulty')
y_test = kdd_t.pop('class')

target = pd.get_dummies(target)
y_test = pd.get_dummies(y_test)

target = target.values
train = kdd.values
test = kdd_t.values
y_test = y_test.values

min_max_scaler = MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)

train,validation_features,target, validation_targets = train_test_split(train, target, test_size=0.30)

train = np.reshape(train, (train.shape[0], 1, train.shape[1]))
validation_features = np.reshape(validation_features, (validation_features.shape[0], 1, validation_features.shape[1]))

print(train.shape)


lstm = lstm_class(alpha=0.1, batch_size=10000, cell_size=120, dropout=0.2,sequence_length=train.shape[2])
lstm_model = lstm.create_model()
checkpoint_p = 'lstm_models/'
result_p = 'lstm_models/'
savemodel_p = 'lstm_models/'
lstm.train(checkpoint_path=checkpoint_p,batch_size=120,model=lstm_model,model_path=savemodel_p, epochs=1000, X_train=train,y_train= target,X_val=validation_features,y_val=validation_targets,result_path=result_p)

#lstm.predict(batch_size=1000,X_test=test,y_test=y_test,model_path=arguments.load_model, result_path=arguments.result_path)


# In[4]:


#lstm_models/training-LSTM-Results-0.99.npy
test = np.reshape(test, (test.shape[0], 1, test.shape[1]))
lstm.predict(batch_size=1000,X_test=test,y_test=y_test,model_path='lstm_models/120-weights.-122.hdf5', result_path=result_p)


# In[ ]:




