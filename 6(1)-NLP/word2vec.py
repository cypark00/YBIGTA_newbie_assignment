import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!

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
        # 구현하세요!
        
        self.pad_token_id = None 

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
        # 구현하세요!
        for epoch in range(num_epochs):
            if self.method == "cbow":
                self._train_cbow(corpus, tokenizer, criterion, optimizer)
            elif self.method == "skipgram":
                self._train_skipgram(corpus, tokenizer, criterion, optimizer)

    def _train_cbow(
        self,
        corpus, tokenizer, criterion, optimizer
    ) -> None:
        # 구현하세요!
        for sentence in corpus:
            token_ids = tokenizer(sentence, add_special_tokens=False)

            # padding token 제거 
            token_ids = [token for token in token_ids if token != self.pad_token_id]

            
            for center_idx in range(self.window_size, len(token_ids) - self.window_size):
                context = (
                    token_ids[center_idx - self.window_size:center_idx] +
                    token_ids[center_idx + 1:center_idx + self.window_size + 1]
                )
        
                target = token_ids[center_idx]
                

                if target == self.pad_token_id or any(token == self.pad_token_id for token in context):
                    continue
                

                context_tensor = torch.tensor(context).to(self.embeddings.weight.device) 
                target_tensor = torch.tensor([target]).to(self.embeddings.weight.device) 
              
                context_vecs = self.embeddings(context_tensor) 
                context_avg = context_vecs.mean(dim=0) 
                
                logits = self.weight(context_avg) 
                
                loss = criterion(logits.unsqueeze(0), target_tensor) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _train_skipgram(
        self,
        corpus: list[str],
        tokenizer,
        criterion,
        optimizer
    ) -> None:
        # 구현하세요!
        for sentence in corpus:
        
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            # padding token 제거
            token_ids = [token for token in token_ids if token != self.pad_token_id]

            
            for center_idx in range(self.window_size, len(token_ids) - self.window_size):
                center = token_ids[center_idx]
                context = (
                    token_ids[center_idx - self.window_size:center_idx] +
                    token_ids[center_idx + 1:center_idx + self.window_size + 1]
                )
                
                if center == self.pad_token_id or any(token == self.pad_token_id for token in context):
                    continue
                
                center_tensor = torch.tensor([center]).to(self.embeddings.weight.device)
                center_vec = self.embeddings(center_tensor).squeeze(0).detach() 

                for ctx in context:
                    context_tensor = torch.tensor([ctx]).to(self.embeddings.weight.device) 
                    logits = self.weight(center_vec) 
                    
                    loss = criterion(logits.unsqueeze(0), context_tensor)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()