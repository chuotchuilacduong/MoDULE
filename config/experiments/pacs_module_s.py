seed: 42
dataset: pacs
data_dir: "./data/PACS"                 
unlearn_setting: class                     
forget_classes: [0]                         
forget_ratio: 0.1                          


model_name: module_resnet50                   
pretrained: true

num_experts: 12                              
gate_k: 4                                   
expert_depth: 2                           
expert_hidden_ratio: 4                        

epochs: 100                                   
batch_size: 128                             
lr: 0.0005                                  
weight_decay: 0.05                           

lambda_sparse: 0.5                           
lambda_balance: 0.5                         
lambda_div: 0.5                            
ema_alpha: 0.9                           