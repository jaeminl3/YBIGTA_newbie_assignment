import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam
from transformers import PreTrainedTokenizer
from typing import Literal
import random

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
    
    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        
        tokenized_corpus = [tokenizer.encode(text, add_special_tokens=False) for text in corpus]
        
        for epoch in range(num_epochs):
            total_loss = 0
            for sentence in tokenized_corpus:
                if len(sentence) < self.window_size * 2 + 1:
                    continue
                
                if self.method == "cbow":
                    loss = self._train_cbow(sentence, criterion, optimizer)
                else:
                    loss = self._train_skipgram(sentence, criterion, optimizer)
                
                total_loss += loss
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    
    def _train_cbow(
        self,
        sentence: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        loss = 0
        for idx in range(self.window_size, len(sentence) - self.window_size):
            context = sentence[idx - self.window_size: idx] + sentence[idx + 1: idx + self.window_size + 1]
            target = sentence[idx]
            
            context_embeddings = self.embeddings(LongTensor(context)).mean(dim=0)
            logits = self.weight(context_embeddings)
            
            target_tensor = LongTensor([target])
            batch_loss = criterion(logits.unsqueeze(0), target_tensor)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            loss += batch_loss.item()
        return loss
    
    def _train_skipgram(
        self,
        sentence: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        loss = 0
        for idx in range(self.window_size, len(sentence) - self.window_size):
            target = sentence[idx]
            context = sentence[idx - self.window_size: idx] + sentence[idx + 1: idx + self.window_size + 1]
            
            target_embedding = self.embeddings(LongTensor([target]))
            logits = self.weight(target_embedding)
            
            context_tensor = LongTensor(context)
            batch_loss = criterion(logits.repeat(len(context), 1), context_tensor)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            loss += batch_loss.item()
        return loss
