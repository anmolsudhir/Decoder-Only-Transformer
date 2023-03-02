# IMPORTING PYTORCH  
import torch
import torch.nn as nn
from torch.nn import functional as F
# import time

# Seed for setting manual_seed()
seed = 1337 # Set it as a random integer to have random seeds

# Setting seed to generate randoms in the model
torch.manual_seed(seed)

# HYPERPARAMETERS
batch_size = 64 # NUmber of tocken array a batch contains
block_size = 256 # Number of token in every token array called block
max_iters = 3000 # Setting number of iteration for training the model
eval_intervals = 500 # To evaluate average loss at every eval_interval iteration 
eval_iters = 200 # Total number of loss calculated to be taken mean of (average loss)
learning_rate = 3e-4 # Learning rate for 'Adam' optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Setting device to be used for the model
n_embd = 384 # Setting the size of embedding table
n_layers = 6 # Number of blocks
n_heads = 6 # Number of head in multi-head self attention
dropout = 0.2 # Fraction of neuron to be dropped when backpropogating

# Importing the text dataset
with open('input.txt', 'r') as f:
  text = f.read()

# Getting a sorted list of all the characters present in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars) # Number of distict characters present in Dataset

# Encoder and Decoder fubctions
stoi = {ch : i for i, ch in enumerate(chars)} # String to Integer encoding
itos = {i : ch for i, ch in enumerate(chars)} # Integer to string decoding
encode = lambda s : [stoi[c] for c in s] # Encode function
decode = lambda l : ''.join([itos[i] for i in l]) # Decode function

# Encoding all the characters present in dataset
data = torch.tensor(encode(text), dtype=torch.long) #Encoded dataset as a tensor

# Splitting Training and Validation data
n = int(0.9*len(data)) # calc 90% of original data
train_data = data[:n] # Training data -> 90%
val_data = data[n:] # Validation data -> 10%

# Loaading data / Getting batchs for training
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size, )) # Randomly generating batch's starting point
  x = torch.stack([data[i:i+block_size] for i in ix]) # Generating context blocks (list of inputs) from randomly generated batch starting point
  y = torch.stack([data[i+1: i+1+block_size] for i in ix]) # Generating Target for context blocks
  # Move to deivce on which model is on
  x = x.to(device)
  y = y.to(device)
  return x, y

@torch.no_grad() # Instructing PyTorch not to run the below function while backpropogating the nn
def estimate_loss():
  out = {} # Output dictionary
  m.eval() # Putting the model in evaluation mode
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters) # Creating tensor of size eval_iter = 200
    for k in range(eval_iters):
      X, Y = get_batch(split) # Getting batches to calculate loss
      logits, loss = m(X, Y) # Calculating prediction (logits) and loss
      losses[k] = loss.item() # Setting loss of kth iteration
    out[split] = losses.mean() # Calculation average loss of training/validation
  m.train() # Putting the model in training mode
  return out

#Creating a class single headed self attention
class SelfAttention(nn.Module):
    def __init__(self,head_size) -> None:
      super().__init__()
      self.head_size = head_size # Getting the head size to create the key, query and value matrix
      self.query = nn.Linear(n_embd, self.head_size, bias = False) # Query -> ()
      self.key = nn.Linear(n_embd, self.head_size, bias = False) # Key 
      self.value = nn.Linear(n_embd, self.head_size, bias = False) # Value
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # A lower triangular matrix for the entries being 1
      self.drop = nn.Dropout(dropout) # Dropout layer
    
    def forward(self, idx):
        B, T, C = idx.shape # Getting the shape of the input
        
        # q, k, v are matrix created by linear transformation of the input idx
        self.q = self.query(idx)
        self.k = self.key(idx)
        self.v = self.value(idx)
        
        # Intermediate matrix 'wei' to store 
        self.wei = self.q @ self.k.transpose(-2, -1)*(C**-0.5) # scaling the matrix multiplication of q and t 
        self.wei = self.wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Ensuring each token considers only the previous(past) token to calculate next token
        self.wei = F.softmax(self.wei, dim=-1) # Applying softmax 
        self.wei = self.drop(self.wei) # Dropping some neurons
        self.out = self.wei @ self.v # Final output
        
        return self.out
    
# Creating multi-headed self attention
class MultiHead(nn.Module):
    def __init__(self, n, size) -> None:
      super().__init__()
      self.proj = nn.Linear(n_embd, n_embd) # A simple transformation layer
      self.heads = nn.ModuleList([SelfAttention(size) for _ in range(n)]) # Creating multiple self-attention heads 
      self.drop = nn.Dropout(dropout) # Adding a dropout layer
      
    def forward(self, idx):
        # Concatinating the outputs of self-attention heads 
        # Transforming the output
        # Adding the dropout
        out = self.drop(self.proj(torch.cat([h(idx) for h in self.heads], dim=-1)))
        return out

# Creating a simple feed forward layer
class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
      super().__init__()
      #  Creating a feed forward network
      self.net = nn.Sequential(
          nn.Linear(n_embd, n_embd*4), # Linear Transforming layer
          nn.ReLU(), # rectified linear unit layer
          nn.Linear(n_embd*4, n_embd), # Projection layer
          nn.Dropout(dropout) # Dropout layer
      )
    
    def forward(self, x):
        return self.net(x) 
    
# Creating a block that will consist of multi headed self attention and a feed forward layer
class Block(nn.Module):
    def __init__(self) -> None:
       super().__init__()
       self.sa_heads = MultiHead(n_heads, n_embd//n_heads) # Creating multiheads of self-attention
       self.ffwd = FeedForward(n_embd) # A feed-forward layer
       # Layer Normalization for better efficiency
       self.ln1 = nn.LayerNorm(n_embd)
       self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # Adding residual connections to multihead head layer
        x = x + self.ffwd(self.ln2(x)) # Adding residual connections to feed-forward layer
        return x

# Creating a Bigram language model class to predict the next token (char)
class GenerativeLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # creating a simple lookup table by embedding tokens of size vacabXvocab
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Token embedding table
    self.positional_encoding = nn.Embedding(block_size, n_embd) # Positional embedding table
    self.lm_head = nn.Linear(n_embd, vocab_size) # Linear transformation at the end
    self.layer_norm = nn.LayerNorm(n_embd) # Layer normalization
    self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)]) # Creating multiple blocks 
    
  # Overriding Forward method of nn.Module class
  def forward(self, idx, target=None):
    # idx and target are of dimensions (B, T) -> (Batches, Times(Block_size))
    # B -> batch_size, T -> block_size
    B, T = idx.shape
    tkn_embd = self.token_embedding_table(idx) # Predicting the next token # Dimensions -> (B, T, C)
                                             # C -> Channel size = n_embd
    pos_embd = self.positional_encoding(torch.arange(T, device=device))
    x = tkn_embd + pos_embd # Broadcast pos_embd(T, C) to (B, T, C) and adds elements-wise
    x = self.blocks(x) # Passing the input through the block layers
    x = self.layer_norm(x) #passing through a later normalization layer
    logits = self.lm_head(x) # Performs a simple linear transformation (B, T, n_embd) -> (B, T, vocab_size)
    
    if target == None :
      loss = None # Setting loss = none if there is no target to evaluate from
    else : 
      B, T, C = logits.shape # Getting predictions shape
      logits = logits.view(B*T, C) # Changing dimensions to calculate loss
      target = target.view(B*T) # Changing Dimentions to calculate loss
      loss = F.cross_entropy(logits, target) # Calculating loss function 
    return logits, loss

  # Generating the next token(character)
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond) # calling forward method of the model
      logits = logits[:, -1, :] # Taking the next token -> (B, C)
      probs = F.softmax(logits, dim = 1) # Calculating the probabilities of next token
      idx_next = torch.multinomial(probs, num_samples=1) # -> (B, 1)
      idx = torch.cat((idx, idx_next), dim = 1) # Concatinating idx and idx_next
    return idx

# Creating Bigram model
model = GenerativeLanguageModel()
m = model.to(device) # Porting model and model variables to device type

# Creating an optimizer object to optimize model
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

# To write loss to sample_torch_output_loss.txt
loss_text = ""

# Training the created model
for iter in range(max_iters):
  # Evaluating average loss at every eval_interval = 300
  if iter%eval_intervals == 0:
    losses = estimate_loss() # Estimating the average loss
    print(f"step {iter} : train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")
    loss_text = loss_text + f"step {iter} : train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n"

  # creating random batches of data
  xb, yb = get_batch('train')
  logits, loss = m(xb, yb) # Predicting next token and loss
  optimizer.zero_grad(set_to_none = True)
  loss.backward() # Backpropogating
  optimizer.step() # Optimizing model's parameters

# Writing output loss
with open('sample_torch_output_loss.txt', 'w') as file:
  file.write(loss_text)

# Creating initial context
context = torch.zeros((1, 1), dtype = torch.long, device = device)

# Writing output to sample_torch_output.txt
with open('sample_torch_output.txt', 'w') as file:
  # Generating tokens and to write to output
  file.write((decode(m.generate(context, max_new_tokens=50000)[0].tolist())))

# Saving the trained model
file_name = 'sample_torch_gpt_model.pt'
torch.save(m.state_dict(), f"./{file_name}")