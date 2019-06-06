# Twitter-Occupation-Prediction
Code and data accompanying paper "Twitter Homophily: Network Based Prediction of User’s Occupation" to be appearing at ACL 2019.


## Dataset
- Twi_data folder contains the training/development/test sets split
- Twi_data folder contains the processed dataset used in GCN model and Deep Walk model, extracted in around February 2018. 
- The dataset is processed from collected Twitter ego-network for a sample of Twitter users whose occupational classes are labeled. Please refer to the paper for collection and processing details.
- Due to privacy concerns, we are not releasing raw Twitter network with Bio. <br>


### Statistics
- Total number of edges: 586303
- Total number of main users (with real labels): 4557
- Total number of users (including main users): 34603


## Code 
- src folder contains code for running GCN model on the processed dataset.
- To execute, please `cd src` to navigate to src folder and then `python train_model.py`. 


### Data Processing Script
We include the jupyter notebook that process the raw network into a densenly connected network which is then used with GCN and DeepWalk.


## Requirements
- 'python=3.6'
- 'torch'
- 'scipy' 
- 'numpy'


## Acknowledgments
The code for GCN model is forked and modified from https://github.com/tkipf/gcn


