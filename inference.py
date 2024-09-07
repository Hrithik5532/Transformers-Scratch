import torch
from transformers import GPT2Tokenizer
from Llama3 import LLaMA3Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = LLaMA3Model(
        checkpoint['vocab_size'],
        checkpoint['d_model'],
        checkpoint['num_layers'],
        checkpoint['num_heads'],
        checkpoint['d_ff'],
        checkpoint['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# Load the model
model_path = 'llama3_model.pth'
model = load_model(model_path)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(model, tokenizer, prompt)
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}")

# Interactive prompt
while True:
    user_prompt = input("Enter a prompt (or 'quit' to exit): ")
    if user_prompt.lower() == 'quit':
        break
    generated_text = generate_text(model, tokenizer, user_prompt)
    print(f"Generated text: {generated_text}")