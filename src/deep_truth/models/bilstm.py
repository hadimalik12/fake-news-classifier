"""Length-aware bidirectional LSTM for statement classification."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, embeddings: torch.Tensor, hidden_size: int, dropout: float,
                 num_labels: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings.clone(), freeze=False, padding_idx=padding_idx,
        )
        self.encoder = nn.LSTM(embeddings.shape[1], hidden_size,
                               batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if torch.any(lengths < 1) or torch.any(lengths > input_ids.shape[1]):
            raise ValueError("Sequence lengths must be between 1 and the padded width.")
        packed = pack_padded_sequence(
            self.embedding(input_ids), lengths.detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))
