import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(x1 * F.silu(x2))

class GroupedMultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(GroupedMultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)  # Changed to project to full d_model
        self.v_proj = nn.Linear(d_model, d_model)  # Changed to project to full d_model
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rotary_pos_enc = nn.Parameter(torch.zeros(1, 1, self.head_dim))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Rotary Positional Encodings
        q_rotary = q + self.rotary_pos_enc
        
        # Compute attention
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q_rotary, k)
        attn_weights = attn_weights / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)
class LLaMA3Layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(LLaMA3Layer, self).__init__()
        self.self_attn = GroupedMultiQueryAttention(d_model, num_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class LLaMA3Model(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(LLaMA3Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([LLaMA3Layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.fc(x)
        return logits

# # Initialize the model
# vocab_size = tokenizer.vocab_size
# d_model = 768
# num_layers = 12
# num_heads = 12
# d_ff = 3072
# dropout = 0.1

# model = LLaMA3Model(vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
# print(f"Model's embedding layer weight shape: {model.embeddings.weight.shape}")
