from datasets import load_dataset
from transformers import GPT2Tokenizer

def prepare_data(sample_size=40000):
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Select samples for training, validation, and testing
    train_dataset = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
    validation_dataset = dataset['validation'].select(range(min(sample_size, len(dataset['validation']))))
    test_dataset = dataset['test'].select(range(min(sample_size, len(dataset['test']))))

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Set format to PyTorch tensors
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return train_dataset, val_dataset, test_dataset, tokenizer
