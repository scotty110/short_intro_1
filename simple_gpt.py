import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_from_disk
from torch.amp import autocast, GradScaler

# ----------------------------
# Scaled Dot-Product Attention
# ----------------------------
class scaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # Apply optional mask (e.g., causal mask)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        # Normalize scores into probabilities
        attn_weights = F.softmax(scores, dim=-1)
        # Weighted sum of values
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# ----------------------------
# Multi-Head Attention Layer
# ----------------------------
class multiHead(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        # Linear projections for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = scaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Project inputs and split into multiple heads
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        # Adjust mask dimensions for broadcasting
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # Apply scaled dot-product attention
        x, _ = self.attention(q, k, v, mask=mask)

        # Concatenate heads and project back
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return self.fc(x)

# ----------------------------
# Position-wise Feed Forward
# ----------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# ----------------------------
# Decoder Block
# ----------------------------
class decoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = multiHead(d_model, n_head)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention with residual connection
        x = x + self.dropout(self.self_attn(self.norm1(x), x, x, mask))
        # Feed-forward network with residual connection
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

# ----------------------------
# Transformer Model (Decoder-only)
# ----------------------------
class simpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            decoder(d_model, n_head, d_ff, dropout) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    # Causal mask to prevent attention to future tokens
    def _generate_causal_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, src):
        batch_size, seq_len = src.shape
        device = src.device

        # Create position IDs
        pos = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

        # Apply embeddings
        tok_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        # Generate mask for causal decoding
        src_mask = self._generate_causal_mask(seq_len, device)

        # Pass through stacked decoder layers
        for layer in self.layers:
            x = layer(x, src_mask)

        # Project to vocabulary logits
        return self.fc_out(x)

# ----------------------------
# Training Script
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train simpleTransformer on a tokenized dataset")
    parser.add_argument("--tokenized_dataset_path", required=True, help="Path to the tokenized dataset saved with save_to_disk")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 256000
    max_seq_len = 512
    d_model = 128
    n_head = 4
    d_ff = 256
    n_layers = 2
    batch_size = 12
    learning_rate = 1e-4

    if device.type == 'cpu':
        print("CUDA not available, exiting.")
        exit()

    model = simpleTransformer(vocab_size, d_model, n_head, d_ff, n_layers, max_seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')

    ds = load_from_disk(args.tokenized_dataset_path)

    print(f"Starting mixed-precision training on {device}...")
    torch.cuda.synchronize()
    start_time = time.time()

    model.train()
    processed_batches = 0
    total_loss = 0.0

    for batch_data in ds.iter(batch_size=batch_size):
        input_ids = torch.tensor(batch_data['input_ids'], dtype=torch.long, device=device)
        labels = input_ids.clone()
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
            labels = labels[:, :max_seq_len]

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input_ids)
            loss = loss_fn(output.view(-1, vocab_size), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        processed_batches += 1

    torch.cuda.synchronize()
    end_time = time.time()

    avg_loss = total_loss / processed_batches
    total_time = end_time - start_time

    print(f"Processed {processed_batches} batches in {total_time:.4f} seconds.")
    print(f"Final average loss: {avg_loss:.4f}")