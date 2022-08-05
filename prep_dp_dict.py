import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import selfies as sf

def error_smiles_removal(dframe1):
   error_line=[]
   for mol_idx in range(dframe1.shape[0]):
      try:
        mol = Chem.MolFromSmiles(dframe1.loc[mol_idx, 'SMILES'])
        #fingerprint = MACCSkeys.GenMACCSKeys(mol)
      except:
        print("Error smiles", mol_idx)
        error_line.append(mol_idx)
        continue
   if error_line != []:
      print(dframe1.iloc[error_line,:] )
   dframe1=dframe1.drop(error_line).reset_index(drop=True) 
   
   return dframe1

def error_selfies_removal(df):
   error_line=[]
   for mol_idx in range(df.shape[0]):
      try:
        comp = df.loc[mol_idx,'SMILES']
        selfies_x = sf.encoder(comp)
      except:
        print("Error selfies", mol_idx)
        error_line.append(mol_idx)

   if error_line != []:
      print(df.iloc[error_line,:] ) 
   df=df.drop(error_line).reset_index(drop=True) 
   
   return df


def sep_word(data, num):
    res = []   
    for i in range(len(data)):
        res.append([data[i][j: j+ num] for j in range(len(data[i])-num+1)])        
    return res

def smiles_to_selfies(compound_list) :
   compound_sf=[]
   for comp in compound_list:
      compound_sf.append(sf.encoder(comp))
   #print('compound_sf:', compound_sf)

   compound_all =[]
   for compound in compound_sf:
      x = compound.replace("][", "],[")
      y = x.split(',')
      compound_all.append(y) 
   #print('compound_all;', compound_all)
   return compound_all
   
def kmer_selfies(compound_all, kmer) :
   sel_kmer= []
   for comp in compound_all :
      if len(comp)-kmer +1 > 0:
         s=[]
         for j in range(len(comp)-kmer+1) :
            s.append(comp[j:j+kmer])
     
         sel=[]
         for x in s :
            xx=''.join(x)
            sel.append(xx)
         print('kmer-selfies={}'.format(sel)) 
         sel_kmer.append(sel)      
      else :
         sel_kmer.append([])

   return sel_kmer

def df_prep(infile1, infile2, infile3):
    num =[0]*6
    df1 = pd.read_csv(infile1, sep=',') #CV
    num[0] = df1.shape[0]
    df1 = error_smiles_removal(df1)
    df1 = error_selfies_removal(df1) 
    num[3] = df1.shape[0]
    df2 = pd.read_csv(infile2, sep=',') #test
    num[1] = df2.shape[0]
    df2 = error_smiles_removal(df2) 
    df2 = error_selfies_removal(df2)
    num[4] = df2.shape[0]
    df3 = pd.read_csv(infile3, sep=',') #test
    num[2] = df3.shape[0]
    df3 = error_smiles_removal(df3)
    df3 = error_selfies_removal(df3)
    num[5] = df3.shape[0]
    return df1, df2, df3, num  

def pickle_save(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def pickle_read(path):
    with open(path, "rb") as f:
        res = pickle.load(f)      
    return res
    
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data    


