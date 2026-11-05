from typing import Tuple, List, Callable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CategoricalDataset(Dataset):
    def __init__(
            self, 
            dataframe: pd.DataFrame, 
            mask_index: List[int],
            mask_probability: float = 0.2
        ):
        self.dataframe = dataframe
        self.mask_index = mask_index
        self.mask_probability = mask_probability
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx].copy()
        mask = np.random.rand(len(row)) < self.mask_probability
        masked_row = row.where(~mask, other=self.mask_index)
        return torch.tensor(masked_row.values, dtype=torch.long), \
            torch.tensor(row.values, dtype=torch.long)


class EmbeddingNetwork(nn.Module):
    def __init__(
            self, 
            num_categories: List[int], 
            embedding_dim: int,
        ):
        super().__init__()
        num_attributes = len(num_categories)
        
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories + 1, embedding_dim) 
             for categories in num_categories]
        )
        self.fc1 = nn.Linear(num_attributes * embedding_dim, 128)
        self.fc2 = nn.Linear(128, num_attributes * embedding_dim)
        self.output_layers = nn.ModuleList(
            [nn.Linear(embedding_dim, categories+1) 
             for categories in num_categories]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [embed(x[:,i]) for i, embed in enumerate(self.embeddings)]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.split(self.embeddings[0].embedding_dim, dim=1)
        x = [output_layer(sub_x) 
             for sub_x, output_layer in zip(x, self.output_layers)]
        return x
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = [embed(x[:,i]) for i, embed in enumerate(self.embeddings)]
        x = torch.cat(x, dim=1)
        return x
    
    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.split(self.embeddings[0].embedding_dim, dim=1)
        x = [output_layer(sub_x) 
             for sub_x, output_layer in zip(x, self.output_layers)]
        decoded_data = [torch.argmax(output, dim=1) for output in x]
        return torch.stack(decoded_data, dim=1)


def train_embedding(
        model: nn.Module, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int,
        save_epochs: int = 10,
        save_path: str = "embedding_model.pt"
    ) -> List[float]:
    model.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = sum(
                F.cross_entropy(output, target) 
                for output, target in zip(outputs, targets.T)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1} | Loss: {total_loss/len(dataloader)}')
        losses.append(total_loss/len(dataloader))

        if (epoch+1) % save_epochs == 0:
            torch.save(model.state_dict(), save_path)
    return losses