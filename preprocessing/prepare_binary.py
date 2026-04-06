import pickle
import gzip
import numpy as np
from tqdm import tqdm

# Load the prepped data
with gzip.open("battery_prepped.pkl.gz", "rb") as f:
    data = pickle.load(f)

# 1. Create a single long string of all CIFs
# We use the <|endoftext|> token to separate them
full_text = ""
for _, cif_str in tqdm(data, desc="Joining strings"):
    full_text += cif_str + "\n<|endoftext|>\n"

# 2. Character-level tokenization (Standard for CrystaLLM)
chars = sorted(list(set(full_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s] 

# 3. Split into Train (95%) and Val (5%)
n = len(full_text)
train_data = full_text[:int(n*0.95)]
val_data = full_text[int(n*0.95):]

# 4. Export to Binary
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Save the files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')

# Save the meta information (the "dictionary")
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("Successfully created train.bin, val.bin, and meta.pkl!")