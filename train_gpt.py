# train_gpt.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# ======================
# CONFIG
# ======================
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
BLOCK_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LR = 3e-4
MODEL_PATH = "integration_gpt_v1.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD DATA
# ======================
with open("data_english.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# ======================
# BATCHING
# ======================
def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

# ======================
# GPT MODEL
# ======================
class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                batch_first=True
            )
            for _ in range(NUM_LAYERS)
        ])

        self.ln = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=device)

        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(pos)

        x = tok_emb + pos_emb

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln(x)
        logits = self.head(x)

        return logits

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# ======================
# TRAINING
# ======================
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ======================
# SAVE SINGLE FILE
# ======================
torch.save({
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "block_size": BLOCK_SIZE
    }
}, MODEL_PATH)

print("✅ Mini GPT saved as integration_gpt_v1.pth")