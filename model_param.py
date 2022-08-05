def model_param_list():
    model_params = {}
    model_params['out_path'] = ''
    model_params['input_dim_drug'] = 23532
    model_params['input_dim_target'] = 16693
    model_params['train_epoch'] = 13
    model_params['max_drug_seq'] = 50  
    model_params['max_protein_seq'] = 545  
    model_params['emb_size'] = 384 
    model_params['dropout_rate'] = 0.1

    #training
    model_params['training_batch_size'] = 128
    model_params['validation_batch_size'] = 32
    model_params['lr'] = 0.001
    model_params['early_stopping'] = 20
    model_params['max_epoch'] = 50
    model_params['thresh'] = 0.5
    model_params['stopping_met'] = "loss" 

    #DenseNet
    model_params['scale_down_ratio'] = 0.25
    model_params['growth_rate'] = 20
    model_params['transition_rate'] = 0.5
    model_params['num_dense_blocks'] = 4
    model_params['kernal_dense_size'] = 3

    model_params["feature_size"] = model_params['emb_size']  
    model_params["kernel_size"] = 5
    model_params["stride"] = 1
    model_params["n_heads"] = 4
    model_params["d_dim"] = 32 
    model_params["feature"] = 128
    model_params["pooling_dropout"] =0.5
    model_params["linear_dropout"] = 0.3 

    # Encoder
    model_params['intermediate_size'] = 1536
    model_params['num_attention_heads'] = 12
    model_params['attention_probs_dropout_prob'] = 0.1
    model_params['hidden_dropout_prob'] = 0.1
    model_params['flat_dim'] = 78192

    return model_params


