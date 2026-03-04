# inference.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# CONFIGURATION (Must Match train.py)
# ===============================
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
MODEL_PATH = "integration_llm_v1.pth"
MAX_GENERATION_LENGTH = 50
TEMPERATURE = 0.8
TOP_K = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# MODEL DEFINITION (Same as train.py)
# ===============================
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.fc = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)


# ===============================
# LOAD MODEL
# ===============================
checkpoint = torch.load(MODEL_PATH, map_location=device)

vocab = checkpoint['vocab']
idx2word = {v: k for k, v in vocab.items()}

model = SimpleTransformer(len(vocab)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ Model Loaded Successfully\n")


# ===============================
# SAFE TOKEN ENCODING
# ===============================
def encode(prompt):
    tokens = []
    for word in prompt.split():
        if word in vocab:
            tokens.append(vocab[word])
    return tokens


def decode(tokens):
    words = []
    for token in tokens:
        if token in idx2word:
            words.append(idx2word[token])
    return " ".join(words)


# ===============================
# TOP-K SAMPLING FUNCTION
# ===============================
def sample_next_token(logits):
    logits = logits / TEMPERATURE
    probabilities = F.softmax(logits, dim=-1)

    top_k_probs, top_k_indices = torch.topk(probabilities, TOP_K)
    top_k_probs = top_k_probs / torch.sum(top_k_probs)

    next_token = torch.multinomial(top_k_probs, 1)
    return top_k_indices[next_token].item()


# ===============================
# TEXT GENERATION
# ===============================
def generate_text(prompt):
    tokens = encode(prompt)

    if len(tokens) == 0:
        return "⚠️ No known tokens found in vocabulary."

    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(MAX_GENERATION_LENGTH):

        with torch.no_grad():
            output = model(input_tensor)

        next_token_logits = output[0, -1, :]
        next_token = sample_next_token(next_token_logits)

        tokens.append(next_token)
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    return decode(tokens)


# ===============================
# INTERACTIVE LOOP
# ===============================
if __name__ == "__main__":

    print("🔹 Internal IntegrationLLM-v1")
    print("Type 'exit' to quit\n")

    while True:
        prompt = input("Ask integration question: ")

        if prompt.lower() == "exit":
            break

        response = generate_text(prompt)
        print("\n🤖 Model Response:\n")
        print(response)
        print("\n" + "-" * 60 + "\n")