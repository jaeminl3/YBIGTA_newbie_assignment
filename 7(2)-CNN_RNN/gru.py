import torch
from torch import nn, Tensor

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        combined = torch.cat([x, h], dim=-1)
        
        z = torch.sigmoid(self.W_z(combined))  # Update gate
        r = torch.sigmoid(self.W_r(combined))  # Reset gate
        
        combined_reset = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_reset))  # Candidate hidden state
        
        h_next = (1 - z) * h + z * h_tilde  # Final hidden state
        return h_next

class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_len, _ = inputs.shape
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        outputs = []
        for t in range(seq_len):
            h = self.cell(inputs[:, t, :], h)
            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # [batch_size, sequence_length, hidden_size]

        return outputs.mean(dim=1)  # 🔥 Mean pooling 적용

