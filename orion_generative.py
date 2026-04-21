import numpy as np
import os
import glob
import collections
os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin" # path for redundancy
try: # try to use CuPy for gpu accel, otherwise use NumPy for cpu
    import cupy as cp
    cupy_available = True
except (ImportError, FileNotFoundError):
    cp = None
    cupy_available = False
    if ImportError:
        Warning("CuPy library not found. GPU acceleration will be disabled.")
    elif FileNotFoundError:
        Warning("CuPy installation found but no compatible GPU detected. GPU acceleration will be disabled.")

# Configurationably cool variables
data_type = 'multiple' # 'single' or 'multiple'
save_iter = 10000 # amount a training iteration between saving checkpoints
tokenidcount = 256 # 256 for byte-level BPE, can be higher if you want to start with more base tokens (like characters)
priority = 'gpu' # 'cpu' or 'gpu'

# Set the array library based on priority
if priority == 'gpu' and cupy_available:
    xp = cp
    print("Using CuPy for GPU acceleration")
elif priority == 'gpu' and not cupy_available:
    xp = np
    print("CuPy not available, falling back to NumPy")
else:
    xp = np
    print("Using NumPy for CPU computation")



class BPETokenizer: #tokenizer
    def __init__(self):
        self.vocab = {} # Maps token_id -> bytes
        self.merges = {} # Maps (byte1, byte2) -> new_byte
        self.inverse_vocab = {} # Maps bytes -> token_id

    def get_stats(self, ids):
        """Counts frequency of adjacent pairs."""
        counts = collections.defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge_ids(self, ids, pair, idx):
        """Replaces all occurrences of 'pair' with new token 'idx'."""
        newids = []
        i = 0
        while i < len(ids):
            # If not at the last element and matches the pair
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, vocab_size=5000, verbose=True):
        """Learns the BPE merges from raw text."""
        # 1. Start with raw bytes (0-tokenidcount)
        # This ensures we can handle ANY character, even emojis
        ids = list(text.encode("utf-8")) # 67 :D as the kids say
        num_merges = vocab_size - tokenidcount
        
        # Initialize base vocab (0-tokenidcount)
        self.merges = {} 
        self.vocab = {idx: bytes([idx]) for idx in range(tokenidcount)}
        
        if verbose: print(f"Training BPE... Goal: {vocab_size} tokens")

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
                
            # find most frequent pair
            pair = max(stats, key=stats.get)
            
            # create new token ID
            idx = tokenidcount + i
            
            # record the merge rule
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # apply merge to the training data
            ids = self.merge_ids(ids, pair, idx)
            
            if verbose and (i+1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {self.vocab[idx]} (Found {stats[pair]} times)")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        return ids

    def encode(self, text):
        """converts string to list of integers"""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            # find the lowest-index pair that is mergable (order matters!)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # no more merges possible
                
            idx = self.merges[pair]
            ids = self.merge_ids(ids, pair, idx)
        return ids

    def decode(self, ids):
        """Converts List of Integers -> String"""
        # concatenate all byte sequences and decode utf-8
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")
    


# --- BPE SETUP ---
tokenizer = BPETokenizer()
# Load ALL text data into one giant string for training the tokenizer
print("Reading files for BPE training...")
full_text_data = ""
files = sorted(glob.glob(os.path.join('training_data', 'text', '*.txt')))
for filename in files:
    with open(filename, 'r', encoding='utf-8') as f:
        full_text_data += f.read() + "\n"

# target_vocab_size: higher = smarter but more RAM
target_vocab_size = 2000 # 2000 good for testing, 5000+ better
tokenizer.train(full_text_data, vocab_size=target_vocab_size)
# updating globals
# Use the tokenizer's encode method so the same rules apply everywhere
data = tokenizer.encode(full_text_data)
vocab_size = len(tokenizer.vocab)
print(f"Data tokenized! Original chars: {len(full_text_data)} -> Tokens: {len(data)}")



# Hyperparameters
hidden_size = 1000 # Size of the "Memory" (neurons)
seq_length = 30 # How many steps to look back
learning_rate = 1e-2
"""
to calculate parameter size, heres a formula
(neurons squared) + (neurons * vocab_size) + (vocab_size * neurons) + (neurons) + (vocab_size)
"""



# Load brain weights or make some if they dont exist
def load_checkpoint(filename="my_rnn_model.npz"):
    """Loads a frozen brain into the current model."""
    global Wxh, Whh, Why, bh, by
    try:
        data = np.load(filename)
        Wxh = xp.array(data['Wxh'])
        Whh = xp.array(data['Whh'])
        Why = xp.array(data['Why'])
        bh = xp.array(data['bh'])
        by = xp.array(data['by'])
        print(f"Successfully loaded weights from {filename}")
        return True
    except FileNotFoundError:
        Warning(f"No checkpoint found at {filename}. Starting from scratch.")
        # Model parameters (weights)
        # Wxh: Input -> Hidden, Whh: Hidden -> Hidden, Why: Hidden -> Output
        Wxh = xp.random.randn(hidden_size, vocab_size) * 0.01
        Whh = xp.random.randn(hidden_size, hidden_size) * 0.01
        Why = xp.random.randn(vocab_size, hidden_size) * 0.01
        bh = xp.zeros((hidden_size, 1)) # Hidden bias
        by = xp.zeros((vocab_size, 1))  # Output bias

def save_checkpoint(filename="my_rnn_model.npz"):
    """Saves the current state of the 'brain' to a file."""
    print(f"Saving brain to {filename}...")
    # Convert to NumPy arrays for saving
    if priority == 'gpu' and cupy_available:
        np.savez(filename, 
            Wxh=cp.asnumpy(Wxh),
            Whh=cp.asnumpy(Whh),
            Why=cp.asnumpy(Why),
            bh=cp.asnumpy(bh),
            by=cp.asnumpy(by))
    else:
        np.savez(filename, Wxh=Wxh, Whh=Whh, Why=Why, bh=bh, by=by)



# calculate loss, gradients, and last hidden state for a sequence of inputs and targets
def lossFun(inputs, targets, hprev):
    """
    Runs the RNN forward and backward to calculate gradients.
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = xp.copy(hprev)
    loss = 0
    
    # --- 1. Forward Pass (Reading & Generating) ---
    for t in range(len(inputs)):
        xs[t] = xp.zeros((vocab_size,1))
        xs[t][inputs[t]] = 1 # One-hot encoding
        
        # THE CORE FORMULA: New Memory = tanh(Input + Old Memory)
        hs[t] = xp.tanh(xp.dot(Wxh, xs[t]) + xp.dot(Whh, hs[t-1]) + bh)
        
        # Prediction
        ys[t] = xp.dot(Why, hs[t]) + by
        try:
            ps[t] = xp.exp(ys[t]) / xp.sum(xp.exp(ys[t])) # Softmax probability
        except OverflowError: # if too big
            shifted = ys[t] - xp.max(ys[t])
            exp_scores = xp.exp(shifted)
            ps[t] = exp_scores / xp.sum(exp_scores)

        loss += -xp.log(ps[t][targets[t],0]) # Cross-entropy loss

    # --- 2. Backward Pass (Backprop through Time) ---
    dWxh, dWhh, dWhy = xp.zeros_like(Wxh), xp.zeros_like(Whh), xp.zeros_like(Why)
    dbh, dby = xp.zeros_like(bh), xp.zeros_like(by)
    dhnext = xp.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = xp.copy(ps[t])
        dy[targets[t]] -= 1 # Backprop into y
        dWhy += xp.dot(dy, hs[t].T)
        dby += dy
        
        dh = xp.dot(Why.T, dy) + dhnext # Backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # Backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += xp.dot(dhraw, xs[t].T)
        dWhh += xp.dot(dhraw, hs[t-1].T)
        dhnext = xp.dot(Whh.T, dhraw)
        
    # Clip gradients to prevent explosion (Common RNN problem)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        xp.clip(dparam, -5, 5, out=dparam)
        
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]



# --- training loop ---
load_checkpoint("my_rnn_model.npz") 

n, p = 0, 0
hprev = xp.zeros((hidden_size, 1)) #initial hidden state
mWxh, mWhh, mWhy = xp.zeros_like(Wxh), xp.zeros_like(Whh), xp.zeros_like(Why)
mbh, mby = xp.zeros_like(bh), xp.zeros_like(by)

while True:
    # Prepare inputs sweep from left to right in steps seq_length long
    if p + seq_length + 1 >= len(data) or n == 0: 
        hprev = xp.zeros((hidden_size, 1)) # Reset memory at start of file
        p = 0 # Reset pointer
        
    inputs = data[p:p+seq_length]
    targets = data[p+1:p+seq_length+1]

    # Forward, Backward, Loss
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    
    # Parameter Update (Adagrad optimizer - better than simple SGD)
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                  [dWxh, dWhh, dWhy, dbh, dby], 
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / xp.sqrt(mem + 1e-8)

    # progress every n%x iterations
    if n % 1000 == 0:
        print(f'iter {n}, loss: {loss:.4f}')
        
        # Sample from the model to see what it says
        seed_ix = inputs[0]
        x = xp.zeros((vocab_size, 1))
        x[seed_ix] = 1
        h = hprev
        predicted_ids = [seed_ix]
        for i in range(200):
            h = xp.tanh(xp.dot(Wxh, x) + xp.dot(Whh, h) + bh)
            y = xp.dot(Why, h) + by

            shifted_y = y - xp.max(y)
            exp_scores = xp.exp(shifted_y)
            p_dist = exp_scores / xp.sum(exp_scores)

            # convert to NumPy for random.choice since CuPy doesn't have it
            if priority == 'gpu' and cupy_available:
                p_dist_np = cp.asnumpy(p_dist.ravel())
            else:
                p_dist_np = p_dist.ravel()
            ix = np.random.choice(range(vocab_size), p=p_dist_np)
            x = xp.zeros((vocab_size, 1))
            x[ix] = 1
            predicted_ids.append(ix)

        print('----\n' + tokenizer.decode(predicted_ids) + '\n----')
    
    if n % save_iter == 0: # save
        save_checkpoint("my_rnn_model.npz")
        
    p += seq_length
    n += 1
