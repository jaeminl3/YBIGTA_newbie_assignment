from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_texts = dataset["train"]["verse_text"]
    val_texts = dataset["validation"]["verse_text"]
    corpus.extend(train_texts)
    corpus.extend(val_texts)
    return corpus