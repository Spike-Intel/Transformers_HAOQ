"""
For testing the model to confirm it runs.
"""

import torch
from HAOQ import Config, HAOQLMHeadModel

model = HAOQLMHeadModel(Config())

print("Model Structure:")
print(model, end='\n\n')

# Example usage
batch_size = 32
seq_len = 128

input_seq = torch.randint(0, 50257, (batch_size, seq_len))
output = model(input_seq)
print(f"Output Shape: {output.shape}")  # Should print torch.Size([32, 128, 50257])