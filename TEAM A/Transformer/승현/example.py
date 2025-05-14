import torch
import torch.nn as nn
from transformer import Transformer

def main():
    # Example parameters
    src_vocab_size = 10000  # Source vocabulary size
    tgt_vocab_size = 10000  # Target vocabulary size
    d_model = 512          # Model dimension
    num_heads = 8          # Number of attention heads
    num_encoder_layers = 6 # Number of encoder layers
    num_decoder_layers = 6 # Number of decoder layers
    d_ff = 2048           # Feed-forward dimension
    dropout = 0.1         # Dropout rate
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # Example input
    batch_size = 32
    src_seq_length = 20
    tgt_seq_length = 20
    
    # Create dummy input tensors
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_length))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_length))
    
    # Forward pass
    output = model(src, tgt)
    
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    
    # Example of how to use the model for training
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(src, tgt)
    
    # Reshape output for loss calculation
    output = output.view(-1, tgt_vocab_size)
    target = tgt.view(-1)
    
    # Calculate loss
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main() 