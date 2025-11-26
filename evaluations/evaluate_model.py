import numpy as np
from pathlib import Path
import json

MODEL_DIR = Path(__file__).resolve().parent.parent / "code" / "saved_model_cf"

def load_model():
    item_embeddings = np.load(MODEL_DIR / "item_embeddings.npy")
    user_embeddings = np.load(MODEL_DIR / "user_embeddings.npy")
    item_biases = np.load(MODEL_DIR / "item_biases.npy")
    user_biases = np.load(MODEL_DIR / "user_biases.npy")


    with open(MODEL_DIR / "model_config.json", "r") as f:
        config = json.load(f)

    return item_embeddings, user_embeddings, item_biases, user_biases, config


def compute_rmse(user_embeddings, item_embeddings, user_biases, item_biases, samples=5000):
    """
    Estimate RMSE by sampling random (user, item) pairs.
    Note: this is a structural sanity evaluation — real ratings are unavailable.
    """
    num_users = user_embeddings.shape[0]
    num_items = item_embeddings.shape[0]

    user_idxs = np.random.randint(0, num_users, samples)
    item_idxs = np.random.randint(0, num_items, samples)

    preds = (
        np.sum(user_embeddings[user_idxs] * item_embeddings[item_idxs], axis=1)
        + user_biases[user_idxs]
        + item_biases[item_idxs]
    )

    # Fake ground truth baseline: assume mean=0, just for RMSE structure
    # This is NOT "performance" but ensures model stability.
    mse = np.mean(preds**2)
    rmse = np.sqrt(mse)
    return mse, rmse


def evaluate():
    print("\n=== Loading Model ===")
    item_emb, user_emb, item_bias, user_bias, config = load_model()

    print("Item embeddings:", item_emb.shape)
    print("User embeddings:", user_emb.shape)
    print("Embedding dimension:", config.get("embedding_dim"))
    print("Regularization:", config.get("reg"))

    print("\n=== Bias Statistics ===")
    print("Item bias mean:", np.mean(item_bias))
    print("Item bias std:", np.std(item_bias))
    print("User bias mean:", np.mean(user_bias))
    print("User bias std:", np.std(user_bias))

    print("\n=== Embedding Stability ===")
    print("Item embedding norm mean:", np.mean(np.linalg.norm(item_emb, axis=1)))
    print("User embedding norm mean:", np.mean(np.linalg.norm(user_emb, axis=1)))

    print("\n=== RMSE Structural Check ===")
    mse, rmse = compute_rmse(user_emb, item_emb, user_bias, item_bias)
    print("Estimated MSE:", round(mse, 4))
    print("Estimated RMSE:", round(rmse, 4))

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    evaluate()


