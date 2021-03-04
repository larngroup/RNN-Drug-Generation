# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:47:59 2020

@author: bjpsa
"""

### Vocabulary for the LatentGan

import re
from os import path
import numpy as np
'''
This vocabulary considers each character as a single token
'''
class Vocabulary:
     def __init__(self, file, max_len = 100):
         
         self.max_len = max_len
         self.path = file
         
         if (path.exists(file)):
             with open(file, 'r') as f:  #path = 'Vocab.txt'
                 vocab = f.read().split() #list
             
             #encode
             self.char_to_int = dict()
             for i, char in enumerate(vocab):
                 self.char_to_int[char] = i
              
             #decode
             self.int_to_char = dict()
             for i, char in enumerate(vocab):
                 self.int_to_char[i] = char  
             
             self.vocab_size = len(vocab)
             
         else:
             print('There is no {} file'.format(file))
        
     def update_vocab(self, smiles):
        '''
        Updates da vocabulary using a list of smiles.
        reads a list of smiles and returns the vocabulary(=all the characters 
        in the smiles' list)'''
        
        #regex = '(\[[^\[\]]{1,6}\])'# finds tokens of the format '[x]'
        unique_chars = set()
        for i, sm in enumerate(smiles):
            
            # substituir 'Br' por 'R' e 'Cl' por'L'
            sm = sm.replace('Br', 'R').replace('Cl', 'L')

            for char in sm:
                unique_chars.add(char)
        #adding start 'G' and padding tokens
        unique_chars.add('G')   #start
        unique_chars.add('A')   #padding
        unique_chars = sorted(unique_chars)
        #Saving to file
        with open(self.path, 'w') as f:
            for char in unique_chars:
                f.write(char+"\n")
        
        print('Number of unique characters in the vocabulary: {} '.format(len(unique_chars)))
        
        vocab = sorted(list(unique_chars))
        #encode
        self.char_to_int = dict()
        for i, char in enumerate(vocab):
            self.char_to_int[char] = i
         
        #decode
        self.int_to_char = dict()
        for i, char in enumerate(vocab):
            self.int_to_char[i] = char  
        
        self.vocab_size = len(vocab)
        #return unique_chars


     def tokenize(self, smiles):    #Transforms a List of SMILES Strings into a list of tokenized SMILES (where each tokenized SMILES is in itself a list)
        '''
        Parameters
        ----------
        smiles : List of SMILES Strings
    
        Returns 
        -------
        A List of Lists where each entry corresponds to one original SMILES string
         Each SMILES String becomes a List of tokens where Br is replaced by R,
         Cl is replaced by L and all characters become a single token. A START and
         PADDING token is also added
    
        '''
        list_tok_smiles = []  # to save each tokenized SMILES String
        for smile in smiles:
            #regex = '(\[[^\[\]]{1,6}\])'
            #print(smile)
            smile = smile.replace('Br', 'R').replace('Cl', 'L')
            #smile_chars = re.split(regex, smile)
            smile_tok = []
            smile_tok.append('G')      #adding the START token
            for char in smile:
                smile_tok.append(char)

            #padding         
            if len(smile_tok) < self.max_len:
                 dif = self.max_len - len(smile_tok)
                 [smile_tok.append('A') for _ in range(dif)]
                        
            
            #assert len(smile_tok) == self.max_len 
            #print(smile_tok)
            list_tok_smiles.append(smile_tok)
        return(list_tok_smiles)
    
     def encode(self, tok_smiles):
         '''
             Encodes each tokenized SMILES String in 'tok_smiles'

         Parameters
         ----------
         smiles : TYPE List of Lists
             DESCRIPTION. List of tokenized SMILES (List)

         Returns
         -------
         encoded_smiles : TYPE List of List
             DESCRIPTION. List of encoded SMILES (List)

         '''
         encoded_smiles = []
         for smile in tok_smiles:
             smile_idx = []
             for char in smile:
                 smile_idx.append(self.char_to_int[char])
             
             encoded_smiles.append(smile_idx)
         return encoded_smiles
     
        
     def decode(self, encoded_smiles):
         '''
         Parameters
         ----------
         encoded_smiles : TYPE
             DESCRIPTION.

         Returns
         -------
         smiles : TYPE List of smiles strings
             DESCRIPTION.

         '''
         smiles = []
         for e_smile in encoded_smiles:
             smile_chars = []
             for idx in e_smile:
                 if (self.int_to_char[idx] == 'G'):
                     continue
                 if (self.int_to_char[idx] == 'A'):
                     break
                 
                 smile_chars.append(self.int_to_char[idx])
            
             smile_str = ''.join(smile_chars)
             smile_str = smile_str.replace('R', 'Br').replace('L', 'Cl')
         
             smiles.append(smile_str)
         
         return smiles      #list
     
     
     def one_hot_encoder(self,smiles_list):
        '''
         

         Parameters
         ----------
         smiles_list : TYPE list of tokenized SMILES
             DESCRIPTION.

         Returns
         -------
         smiles_one_hot : TYPE 3d numpy array with shape (total_number_smiles, max_lenght_sequence, vocab_size) ready to give as input to the a LSTM model
             DESCRIPTION.

         '''
        
         #smiles_one_hot = []
        smiles_one_hot = np.zeros((len(smiles_list),self.max_len, self.vocab_size), dtype = np.int8)
        for j, smile in enumerate(smiles_list):
           
           #X = np.zeros((self.max_len, self.vocab_size), dtype=np.int8)
           for i, c in enumerate(smile):
               
               smiles_one_hot[j, i,self.char_to_int[c]] = 1
           #smiles_one_hot.append(X)
           #smiles_one_hot[j] = X
        return smiles_one_hot
            
     def one_hot_decoder(self, smiles_array ):
         '''
        
         Parameters
         ----------
         smiles_array : TYPE 3d numpy array with shape (total_number_smiles, max_lenght_sequence, vocab_size)
             DESCRIPTION.

         Returns
         -------
         encoded_smiles : TYPE a list of numpy arrays with shape (max_length_sequence,)
             DESCRIPTION.

         '''

         encoded_smiles = []
         for i in range(smiles_array.shape[0]):
             enc_smile = np.argmax(smiles_array[i,: ,:], axis = 1)
             encoded_smiles.append(enc_smile)
         return encoded_smiles
         
     def get_target(self, dataX, encode):
         
          '''
          Creates the target for the input dataX

          Parameters
          ----------
          dataX : TYPE 
              DESCRIPTION.

          Returns
          -------
          dataY : TYPE equals dataX but with each entry shifted 1 timestep and with an appended 'A' (padding)
              DESCRIPTION.

          '''
          if encode == 'OHE':
            dataY = np.zeros(shape = dataX.shape, dtype = np.int8)  #ohe
            for i in range(dataX.shape[0]):
                dataY[i,0:-1, :]= dataX[i, 1:, :]
                dataY[i,-1,self.char_to_int["A"]] = 1
              
          elif encode == 'embedding':
              dataY = [line[1:] for line in dataX]        #emb
              for i in range(len(dataY)):
                 dataY[i].append(self.char_to_int['A'])
     
          return dataY

      
         