import numpy as np
import os
import glob
os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin" # path for redundancy
try:
    import cupy as cp
    cupy_available = True
except (ImportError, FileNotFoundError):
    cp = None
    cupy_available = False
    if ImportError:
        print("CuPy library not found. GPU acceleration will be disabled.")
    elif FileNotFoundError:
        print("CuPy installation found but no compatible GPU detected. GPU acceleration will be disabled.")

data_type = 'multiple' # 'single' or 'multiple'
save_iter = 10000
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



# --- file config ---
if data_type == 'multiple':
    data_dir = os.path.join('training_data', 'text')
    files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
    data = ''
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            data += f.read()
            data += '\n' # seperate files
else:    
    with open('training_data/text/sample.txt', 'r', encoding='utf-8') as f:
        data = f.read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Hyperparameters
hidden_size = 1000 # Size of the "Memory" (neurons)
seq_length = 30   # How many steps to look back
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
        print("No checkpoint found. Starting from scratch.")
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
        
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

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
        txt = []
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
            txt.append(ix_to_char[ix])
            
        print('----\n' + "".join(txt) + '\n----')
    
    if n % save_iter == 0: # OOOUU SAVE YAAAA
        save_checkpoint("my_rnn_model.npz")
        
    p += seq_length
    n += 1
