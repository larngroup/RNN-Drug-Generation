# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Input, GRU, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from livelossplot import PlotLossesKerasTF
import os

class Generator:
    def __init__(self,current_path, dataset_name, encode,n_smiles, vocab, max_len, n_layers, units, learning_rate, dropout_rate, activation, epochs, batch_size, optimizer, sampling_t):
        
        self.sampling_temp = sampling_t
        
        self.model = None
        self.encode = encode

        self.vocab = vocab
        self.vocab_size = vocab.vocab_size
        self.emb_dim = int(units/2)
        self.max_len = max_len
        
        self.n_layers = n_layers
        self.units = units
        self.learning_rate = learning_rate
        #self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        
        if optimizer == 'adam':
            self.optimizer = optimizer
        elif optimizer =='adam_clip':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, clipvalue=3)
        elif optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
        elif optimizer == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
            

        self.build()
        folder = "Exp8_Temperature" + "_"+encode+"_"+str(n_smiles)+"_LSTM-"+str(n_layers)+"-"+str(units)+ "-" +str(epochs)+"-"+str(batch_size)+"-"+str(dropout_rate)+"-"+optimizer+ "-"+str(self.emb_dim)+"-"+str(self.sampling_temp)+"\\"
        self.path = current_path +"\\"+ folder

        if os.path.exists(self.path):
            pass
        else:
            os.makedirs(self.path)

        
        
        
        
    def build(self):
        self.model = Sequential()

        if self.encode == 'OHE':
            self.model.add(Input(shape = (self.max_len, self.vocab_size)))
        elif self.encode == 'embedding':  
            self.model.add(Embedding(self.vocab_size, self.emb_dim, input_length = self.max_len))
        
        for i in range(self.n_layers): 
            self.model.add(LSTM(self.units, return_sequences=True))            
            if self.dropout_rate != 0:
                self.model.add(Dropout(self.dropout_rate))

        # for i in range(self.n_layers): 
        #     self.model.add(GRU(self.units, return_sequences=True))            
        #     if self.dropout_rate != 0:
        #         self.model.add(Dropout(self.dropout_rate))

        # for i in range(self.n_layers): 
        #     self.model.add(Bidirectional(LSTM(self.units//2, return_sequences=True))) 
        #     if self.dropout_rate != 0:
        #         self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(units = self.vocab_size, activation = self.activation))
        
        print(self.model.summary())
        
        #compile the model
        if self.encode == 'OHE':
            self.model.compile(optimizer  = self.optimizer, loss = 'categorical_crossentropy') #OHE
        elif self.encode == 'embedding':
            self.model.compile(optimizer = self.optimizer, loss = 'sparse_categorical_crossentropy') #'mse' emb
        

    def load_model(self, path):
        self.model.load_weights(path)

    def fit_model(self, dataX, dataY):
        filename="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        
        
        early_stop = EarlyStopping(monitor = "loss", patience=5)

        
        path = self.path+F"{filename}"
        checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, mode = 'min') 
        callbacks_list = [checkpoint, early_stop]#, PlotLossesKerasTF()]
        results = self.model.fit(dataX, dataY, verbose = 1, epochs = self.epochs, batch_size = self.batch_size, shuffle = True, callbacks = callbacks_list)
        
        #plot
        fig, ax = plt.subplots()
        ax.plot(results.history['loss'])
        ax.set(xlabel='epochs', ylabel = 'loss')
        figure_path = self.path + "Loss_plot.png"
        fig.savefig(figure_path)
        #plt.show()
        last_epoch = early_stop.stopped_epoch
        
        return results, last_epoch
    
    
    def sample_with_temp(self, preds):
        
        """
        #samples an index from a probability array 'preds'
        preds: probabilities of choosing a character
        
        """
       
        preds_ = np.log(preds).astype('float64')/self.sampling_temp
        probs= np.exp(preds_)/np.sum(np.exp(preds_))
        #out = np.random.choice(len(preds), p = probs)
        
        out=np.argmax(np.random.multinomial(1,probs, 1))
        return out
        
    
    def generate(self, start_idx, numb, end_idx):
        """
        Generates new SMILES strings, token by token

        Parameters
        ----------
        start_idx : TYPE int
            DESCRIPTION. starting index, usually the one that corresponds to 'O'
        numb : TYPE
            DESCRIPTION. number of SMILES strings to be generated

        Returns
        -------
        list_seq : TYPE list of list
            DESCRIPTION. A list where each entry is a tokenized SMILES 

        """

        list_seq = []
        if self.encode == 'embedding':
        
            for j in tqdm(range(numb)):
                seq = [start_idx]
                #x = np.reshape(seq, (1, len(seq),1))
                
                for i in range(self.max_len-1):
                    x = np.reshape(seq, (1, len(seq),1))
                    preds = self.predict(x)
                    
                    #sample
                    #index = np.argmax(preds[0][-1])
                    #sample with T
                    index = self.sample_with_temp(preds[0][-1])
                    seq.append(index)
                    if (index) == end_idx:
                        break
                list_seq.append(seq)

        elif self.encode =='OHE':

            for j in tqdm(range(numb)):
                start_idx_oh = np.zeros(self.vocab_size, dtype = np.int8)
                start_idx_oh[start_idx] = 1
                
                seq = [start_idx_oh]
                for i in range(self.max_len-1):
                    
                    x = np.reshape(seq, (1, len(seq), self.vocab_size))
                    preds = self.predict(x)

                    index = self.sample_with_temp(preds[0][-1])
                    aux = np.zeros(self.vocab_size, dtype = np.int8)
                    aux[index] = 1
                    seq.append(aux)
                   
                    if (index) == end_idx:
                        break
                
                list_seq.append(seq)

        return list_seq
        
    def predict(self, input_x):
        preds = self.model.predict(input_x, verbose=1)
        return preds
        

if __name__ == '__main__':
    pass
