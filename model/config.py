

# General 
hidden_size: int = 512
is_cuda =  False
embed_size: int = 300
pointer = True
eps = 1e-31
LAMBDA = 1
learning_rate = 0.001
epochs = 8
max_grad_norm = 2.0
batch_size = 8

encoder_save_name = '../saved_model/' + model_name + '/encoder.pt'
decoder_save_name = '../saved_model/' + model_name + '/decoder.pt'
attention_save_name = '../saved_model/' + model_name + '/attention.pt'
reduce_state_save_name = '../saved_model/' + model_name + '/reduce_state.pt'
losses_path = '../saved_model/' + model_name + '/val_losses.pkl'
log_path = '../runs/' + model_name


# Data
max_vocab_size = 20000