# An Ensemble Learning Framework for Click-through Rate Prediction based on a Reinforcement Learning Algorithm with Parameterized Actions

## RLCP:
This is the code of the paper "An Ensemble Learning Framework for Click-through Rate Prediction based on a Reinforcement Learning Algorithm with Parameterized Actions".

## Instructions:
* Step1 Generate the ctr prediction of the base model(for the ipinyou dataset, the base model code is located in the src\ipinyou_base_model, for the criteo dataset, the base model code is located in the src\criteo_base_model,for the avazu dataset, we use the code from https://github.com/xue-pai/FuxiCTR)

* Step2 Run ./main/hybrid_td3_main_per_v13.py to train the model proposed in this paper.The variable "campaign_id" can be used to specify the dataset for training
