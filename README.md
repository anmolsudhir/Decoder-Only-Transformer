# Decoder-Only Transformer

## Description

This is an attempt to create a decoder-only transformer neural network that generates text that is like the given input text.

This project uses Shakespeare's work to learn and generate characters like his literature

This project uses pytorch library

## Sample Outputs

###### Sample output has been written in to sample_torch_output.txt
    Generated 50,000 new tokens
    Hyperparameters set for this sample generations are : 
        -[x]batch_size = 64 # Number of tocken array a batch contains
        -[x]block_size = 256 # Number of token in every token array called block
        -[x]max_iters = 3000 # Setting number of iteration for training the model
        -[x]eval_intervals = 500 # To evaluate average loss at every eval_interval iteration 
        -[x]eval_iters = 200 # Total number of loss calculated to be taken mean of (average loss)
        -[x]learning_rate = 3e-4 # Learning rate for 'Adam' optimizer
        -[x]device = 'cuda' if torch.cuda.is_available() else 'cpu' # Setting device to be used for the model
        -[x]n_embd = 384 # Setting the size of embedding table
        -[x]n_layers = 6 # Number of blocks
        -[x]n_heads = 6 # Number of head in multi-head self attention
        -[x]dropout = 0.2 # Fraction of neuron to be dropped when backpropogating

###### Sample output loss is written to sample_torch_outpu_loss.txt
    -[x] It contains loss at every 'eval_interval' iteration