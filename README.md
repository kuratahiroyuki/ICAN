# ICAN:Interpretable cross attention network for identifying drug-target protein interaction

## Code
* The code of FCS embedding methods is built based on MolTrans [https://github.com/kexinhuang12345/MolTrans]. Thanks a lot for their code sharing!

## Dataset
* All datasets (DAVIS, BindingDB, BIOSNAP) used in this study are publicly available.

# Environment
 >anaconda 4.11.0  
 >python 3.7.10  
 >pytorch 1.9.0  
 >scikit-learn 1.0.2  
 >RDKit 2020.09.1  
 >gensim 4.0.1  
 >subword-nmt 0.3.8  
    https://pypi.org/project/subword-nmt/  
 >SELFIES  
    https://github.com/aspuru-guzik-group/selfies  
 >Biopython  
    https://biopython.org/docs/1.75/api/Bio.html  
 
# Execution
# 1 Setting directories
Users must make four directories:dataset, ESPF, dictionary, result.

# 2 Construction of dataset and ESPF
## 2-1 Dataset and ESPF
Before simulation, users must input the following dataset files:  
MolTrans-master/dataset/*  
MolTrans-master/ESPF/*  
into the directories of "dataset" and "ESPF" of "ICAN".  
>cp -r  MolTrans-master/dataset/  ICAN/dataset/  
>cp -r  MolTrans-master/ESPF/  ICAN/ESPF/  
"MolTrans-master" is freely available at https://github.com/kexinhuang12345/MolTrans.  
If users use your own dataset, they need to remake the following dictionaries.  

## 2-2 Dictionary and result
The "dictionary" directory includes the dictrionaries that the authors have constructed for the DAVIS, BindingDB, and BIOSNAP datasets.  
After simulation, the simulation results are saved in the "result" directory.

# 3 Simulation
## 3-1 Main simulation
$sh main.sh  
(traintest.py, deep_network.py)  
Users must set iter_num <= 5.
## 3-2 Optional simulation
$sh main.sh  
Users can choose three encoding methods: FCS, SMILES, SELFIES  
Users can choose three datasets: DAVIS, BindingDB, BIOSNAP/full_data  

