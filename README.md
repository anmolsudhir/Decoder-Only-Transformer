# Decoder-Only Transformer

## Description

This is an attempt to create a decoder-only transformer neural network that generates text that is like the given input text.

This project uses Shakespeare's work to learn and generate characters like his literature

This project uses pytorch library

## Sample Outputs

Sample output has been written in to sample_torch_output.txt
    Generated 50,000 new tokens
    Hyperparameters set for this sample generations are : 
        -> batch_size = 64 # Number of tocken array a batch contains
        -> block_size = 256 # Number of token in every token array called block
        -> max_iters = 3000 # Setting number of iteration for training the model
        -> eval_intervals = 500 # To evaluate average loss at every eval_interval iteration 
        -> eval_iters = 200 # Total number of loss calculated to be taken mean of (average loss)
        -> learning_rate = 3e-4 # Learning rate for 'Adam' optimizer
        -> device = 'cuda' if torch.cuda.is_available() else 'cpu' # Setting device to be used for the model
        -> n_embd = 384 # Setting the size of embedding table
        -> n_layers = 6 # Number of blocks
        -> n_heads = 6 # Number of head in multi-head self attention
        -> dropout = 0.2 # Fraction of neuron to be dropped when backpropogating

Sample output loss is written to sample_torch_outpu_loss.txt
    It contains loss at every 'eval_interval' iteration