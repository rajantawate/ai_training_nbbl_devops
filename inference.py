# inference.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Interactive Chat Started. Type 'exit' to quit.\n")

chat_history = ""

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history += f"User: {user_input}\nAssistant: "

    inputs = tokenizer(chat_history, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only latest assistant reply
    reply = response[len(chat_history):].strip()

    print(f"Assistant: {reply}\n")

    chat_history += reply + "\n"