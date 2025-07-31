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
        self.vocab_size = vocab_size 
        self.d_model = d_model

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
        
        tokenized_corpus = [
            tokenizer(sentence, add_special_tokens=False)["input_ids"]
        for sentence in corpus
        ]

        if self.method == "cbow":
            self._train_cbow(tokenized_corpus, criterion, optimizer, num_epochs)
        elif self.method == "skipgram":
            self._train_skipgram(tokenized_corpus, criterion, optimizer, num_epochs)


    def _train_cbow(
        self,
        tokenized_corpus: list[list[int]],
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        num_epochs: int
    ) -> None:
        # 구현하세요!
        for epoch in range(num_epochs):
            total_loss = 0 
            num_batches = 0 

            for tokens in tokenized_corpus:
                for i in range(self.window_size, len(tokens) - self.window_size):
                    context = (
                    tokens[i - self.window_size:i] + tokens[i + 1:i + self.window_size + 1]
                    )
                    target = tokens[i]

                    if not context:
                        continue

                    context_ids = torch.LongTensor(context)  # (2 * window_size,)
                    target_id = torch.LongTensor([target])

                    context_embeds = self.embeddings(context_ids)  # (2*window, d_model)
                    input_embed = context_embeds.mean(dim=0) 

                    logits = self.weight(input_embed)        # (vocab_size,)
                    loss = criterion(logits.unsqueeze(0), target_id)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

            avg_loss = total_loss / (num_batches or 1)
            print(f"[CBOW] Epoch {epoch+1}/{num_epochs} → 평균 loss: {avg_loss:.4f}")



    def _train_skipgram(
        self,
        tokenized_corpus: list[list[int]],
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        num_epochs: int
        # 구현하세요!
    ) -> None:
        # 구현하세요!
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            for tokens in tokenized_corpus:
                for i in range(self.window_size, len(tokens) - self.window_size):
                    center = tokens[i]
                    context = (
                        tokens[i - self.window_size:i] + tokens[i + 1:i + self.window_size + 1]
                    )

                    for target in context:
                        input_id = torch.LongTensor([center])   # 중심 단어
                        target_id = torch.LongTensor([target])  # 주변 단어

                        input_embed = self.embeddings(input_id).squeeze(0)  # (d_model,)
                        logits = self.weight(input_embed)             # (vocab_size,)
                        loss = criterion(logits.unsqueeze(0), target_id)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        num_batches += 1

            avg_loss = total_loss / (num_batches or 1)
            print(f"[Skipgram] Epoch {epoch+1}/{num_epochs} → 평균 loss: {avg_loss:.4f}")

    # 구현하세요!
    