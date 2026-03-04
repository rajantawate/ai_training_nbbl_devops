# inference_gpt.py

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "integration_gpt_v1.pth"
TEMPERATURE = 0.8
TOP_K = 20
MAX_NEW_TOKENS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# LOAD CHECKPOINT
# ============================
checkpoint = torch.load(MODEL_PATH, map_location=device)
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
config = checkpoint["config"]

vocab_size = len(stoi)


# ============================
# GPT MODEL (MUST MATCH TRAIN)
# ============================
class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, config["embed_dim"])
        self.pos_emb = nn.Embedding(config["block_size"], config["embed_dim"])

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config["embed_dim"],
                nhead=config["num_heads"],
                batch_first=True
            )
            for _ in range(config["num_layers"])
        ])

        self.ln = nn.LayerNorm(config["embed_dim"])
        self.head = nn.Linear(config["embed_dim"], vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=device)

        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(pos)

        x = tok_emb + pos_emb

        # Causal mask
        mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()

        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens=MAX_NEW_TOKENS):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -config["block_size"]:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / TEMPERATURE

            # Top-K sampling
            if TOP_K is not None:
                v, ix = torch.topk(logits, TOP_K)
                mask = logits < v[:, [-1]]
                logits[mask] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx


# ============================
# INITIALIZE MODEL
# ============================
model = GPT().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ Mini GPT Loaded Successfully")
print("Type 'exit' to quit\n")


# ============================
# ENCODE / DECODE
# ============================
def encode(text):
    tokens = [stoi[c] for c in text.lower() if c in stoi]
    return torch.tensor([tokens], dtype=torch.long).to(device)


def decode(tokens):
    return "".join([itos[t] for t in tokens])


# ============================
# INTERACTIVE LOOP
# ============================
while True:

    prompt = input("Ask integration question: ")

    if prompt.lower() == "exit":
        break

    if len(prompt.strip()) == 0:
        print("⚠️ Please enter a valid question.\n")
        continue

    input_ids = encode(prompt)

    if input_ids.shape[1] == 0:
        print("⚠️ No recognizable characters in vocabulary.\n")
        continue

    with torch.no_grad():
        output_ids = model.generate(input_ids)

    response = decode(output_ids[0].tolist())

    print("\n🤖 Response:\n")
    print(response)
    print("\n" + "-" * 60 + "\n")