# Improving CTR Prediction Performance Using Ensemble Learning Framework with Reinforcement Learning

## RLCP:
This is the code of the paper "Improving CTR Prediction Performance Using Ensemble Learning Framework with Reinforcement Learning".

## Instructions:

* Step1 The IPinYou dataset used in this paper comes from make-ipinyou-data https://github.com/wnzhang/make-ipinyou-data. Unzip the dataset following the introduction on the above website, and copy the dataset to the ./data directory.
* Step2 Run ./encode/data_.py  to preprocess the raw dataset.
* Step3 Run ./main/pretrain_main.py to pretrain several basic click-through prediction models.
* Step4 Run ./main/hybrid_td3_main_per_v13.py to train the model proposed in this paper.The variable "campaign_id" can be used to specify the dataset for training
