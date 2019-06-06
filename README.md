# Twitter-Occupation-Prediction
Code and data accompanying paper "Twitter Homophily: Network Based Prediction of User’s Occupation" to be appearing at ACL 2019.


## Dataset
- Contains the processed dataset used in GCN model and Deep Walk model, extracted in around February 2018. The dataset is processed from collected Twitter ego-network for a sample of Twitter users whose occupational classes are labeled. Please refer to the paper for collection and processing details.
- Contains the training/development/test sets split


### Statistics
- Total number of edges: 586303
- Total number of main users (with real labels): 4557
- Total number of users (including main users): 34603


## Code 
- Contains code for running GCN model on the processed dataset.
- To execute, please `cd src` to navigate to src folder and then `python train_model.py`. 


## Requirements
- 'python=3.6'
- 'torch'
- 'scipy' 
- 'numpy'


## Acknowledgments
The code for GCN model is forked and modified from https://github.com/tkipf/gcn


