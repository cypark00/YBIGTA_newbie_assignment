# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    corpus: list[str] = []
    for example in dataset:
        verse = example["verse_text"].strip()
        if verse != "":
            corpus.append(verse)
    return corpus