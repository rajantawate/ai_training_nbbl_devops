# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# ===============================
# Configuration
# ===============================
VOCAB_SIZE = 5000
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 100
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = "integration_llm_v1.pth"


# ===============================
# Simple Tokenizer (Basic)
# ===============================
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        words = list(set(text.split()))
        for word in words:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

    def encode(self, text):
        return [self.word2idx[word] for word in text.split() if word in self.word2idx]

    def decode(self, tokens):
        return " ".join([self.idx2word[token] for token in tokens])


# ===============================
# Dataset
# ===============================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer):
        self.tokenizer = tokenizer
        tokens = tokenizer.encode(text)
        self.data = []
        for i in range(len(tokens) - MAX_SEQ_LEN):
            self.data.append(tokens[i:i+MAX_SEQ_LEN+1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


# ===============================
# Transformer Model
# ===============================
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq, batch, embed)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.fc(x)


# ===============================
# Load Data
# ===============================
with open("data_english.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(text_data)

dataset = TextDataset(text_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleTransformer(tokenizer.vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===============================
# Training Loop
# ===============================
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ===============================
# Save Single .pth File
# ===============================
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': tokenizer.word2idx
}, MODEL_PATH)

print("Model saved as integration_llm_v1.pth")