import numpy as np
from pathlib import Path

MODEL_DIR = Path("saved_model")

def load_model():
    item_embeddings = np.load(MODEL_DIR / "item_embeddings.npy")
    item_biases = np.load(MODEL_DIR / "item_biases.npy")
    return item_embeddings, item_biases

def evaluate():
    item_embeddings, item_biases = load_model()
    print("Model loaded for evaluation.")
    print("Embedding shape:", item_embeddings.shape)
    print("Bias shape:", item_biases.shape)

if __name__ == "__main__":
    evaluate()
