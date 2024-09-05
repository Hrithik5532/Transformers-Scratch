import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from preparedata import prepare_data
from Llama3 import LLaMA3Model

# Hyperparameters
num_epochs = 3
batch_size = 8
learning_rate = 5e-5

train_dataset, val_dataset, test_dataset, tokenizer = prepare_data()
# Initialize the model
vocab_size = tokenizer.vocab_size
d_model = 768
num_layers = 12
num_heads = 12
d_ff = 3072
dropout = 0.1

model = LLaMA3Model(vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
model.to(device)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.to(device)

        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}")

    # Evaluation
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            val_loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")
    
import os

    
def save_model(model, tokenizer, output_dir):
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the model
        model.to('cpu')  # Move the model to CPU before saving
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
        
        # Save the tokenizer
        tokenizer.save_pretrained(output_dir)
        
        print(f"Model and tokenizer saved to {output_dir}")
        
        
def load_model(model_class, tokenizer_class, input_dir):
    # Load the tokenizer
    tokenizer = tokenizer_class.from_pretrained(input_dir)
    
    # Initialize the model (you need to know the model's hyperparameters)
    model = model_class(
        vocab_size=tokenizer.vocab_size,
        d_model=768,  # Adjust these parameters as needed
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        dropout=0.1
    )
    
    # Load the model state
    model.load_state_dict(torch.load(os.path.join(input_dir, 'model.pth')))
    
    return model, tokenizer

# After training, save the model
output_dir = 'llama3_model'
save_model(model, tokenizer, output_dir)

# To load the model later (in the same script or a different one):
from Llama3 import LLaMA3Model  # Make sure this import is correct
from transformers import GPT2Tokenizer  # Assuming you're using GPT2Tokenizer

loaded_model, loaded_tokenizer = load_model(LLaMA3Model, GPT2Tokenizer, 'llama3_model')

# Move the loaded model to the desired device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

print("Model loaded successfully!")