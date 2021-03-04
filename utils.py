# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:37:26 2020

@author: bjpsa
"""
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit import DataStructs
from functools import reduce
import matplotlib.pyplot as plt

def validity(smiles_list):
    '''
    Evaluates if the generated SMILES are valid using rdkit
    Parameters
    ----------
    smiles_list : TYPE
        DESCRIPTION. List of Smiles Strings

    Returns
    -------
    valid_smiles : TYPE List 
        DESCRIPTION. list of SMILES strings that were deamened valid
    perc_valid : TYPE
        DESCRIPTION. percentage of valid SMILES strings in the input data

    '''
    
    total = len(smiles_list)
    valid_smiles =[]
    count = 0
    for sm in smiles_list:
        if MolFromSmiles(sm) != None and sm !='':
            valid_smiles.append(sm)
            count = count +1
    perc_valid = count/total*100
    
    return valid_smiles, perc_valid
        
    
def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t*1000,), 1000, 60, 60])


def uniqueness(smiles_list):
    
    unique_smiles = list(set(smiles_list))
    
    return (len(unique_smiles)/len(smiles_list))*100

def diversity(smiles_A,smiles_B = None):
# If you want to compute internal similarity just put the 
# filename_a and the filename_b as 'None'. If you want to compare two sets, 
# write its names properly and it will be computed the Tanimoto distance.
# Note that it is the Tanimoto distance, not Tanimoto similarity. 
 
    td = 0
    print(smiles_A)
    print(smiles_B)
    fps_A = []
    for i, row in enumerate(smiles_A):
        try:
            mol = MolFromSmiles(row)
            fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
        except:
            print('ERROR: Invalid SMILES!')
            
        
    
    if smiles_B == None:
        for ii in range(len(fps_A)):
            for xx in range(len(fps_A)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                td += ts          
        
        if len(fps_A) == 0:
            td = None
        else:
            td = td/len(fps_A)**2
    else:
        fps_B = []
        for j, row in enumerate(smiles_B):
            try:
                mol = MolFromSmiles(row)
                fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!') 
        
        
        for jj in range(len(fps_A)):
            for xx in range(len(fps_B)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                td += ts
        
        if (len(fps_A) == 0 or len(fps_B) == 0):
            td = None
        else:   
            td = td / (len(fps_A)*len(fps_B))
    print("Tanimoto distance: " + str(td))  
    return td

    
    
def display_epochs(epochs, valid, uniq, name):

    #plot
    fig, ax = plt.subplots()
    ax.plot(epochs, valid, label ='Valid')
    ax.plot(epochs, uniq, label = 'Unique')
    ax.set(xlabel='Number of Epochs', ylabel = 'percentage(%)')
    ax.legend()
    #ax.set_ylim(0,100)
    figure_path = name + '.png'
    fig.suptitle('Model: '+name)
    fig.savefig(figure_path)
    #plt.show()
    
    


