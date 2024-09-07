import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn as nn
from preparedata import prepare_data
from Llama3 import LLaMA3Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# Save the model after training
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab_size': vocab_size,
    'd_model': d_model,
    'num_layers': num_layers,
    'num_heads': num_heads,
    'd_ff': d_ff,
    'dropout': dropout
}, 'llama3_model.pth')

print("Model saved successfully!")