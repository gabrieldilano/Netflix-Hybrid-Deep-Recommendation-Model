import numpy as np
import pickle
from pathlib import Path
import json
import sys

MODEL_DIR = Path(__file__).resolve().parent.parent / "code" / "saved_model_cf"


def load_model():
    item_embeddings = np.load(MODEL_DIR / "item_embeddings.npy")
    user_embeddings = np.load(MODEL_DIR / "user_embeddings.npy")
    item_biases = np.load(MODEL_DIR / "item_biases.npy")
    user_biases = np.load(MODEL_DIR / "user_biases.npy")
    with open(MODEL_DIR / "model_config.json", "r") as f:
        config = json.load(f)

    return item_embeddings, user_embeddings, item_biases, user_biases, config


def load_metadata():
    repo_root = Path(__file__).resolve().parent.parent / "code"
    cbf_dir = repo_root / "saved_model_cbf"
    cf_dir = repo_root / "saved_model_cf"

    metadata_path = cbf_dir / "movies_metadata.pkl"
    mappings_path = cf_dir / "mappings.pkl" if (cf_dir / "mappings.pkl").exists() else cbf_dir / "mappings.pkl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"movies_metadata.pkl not found at {metadata_path}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"mappings.pkl not found at {mappings_path}")




def recommend_for_user(user_id, top_k=10):
    item_emb, user_emb, item_bias, user_bias, config = load_model()

    try:
        metadata, mappings = load_metadata()
    except Exception as e:
        print("Warning: failed to load metadata/mappings:", e)
        # Build a minimal fallback metadata + mappings so recommendations can still run.
        try:
            import pandas as pd

            num_items = int(item_emb.shape[0])
            metadata = pd.DataFrame({"title": [f"Movie {i}" for i in range(num_items)]})
            metadata.index = range(num_items)
        except Exception:
            metadata = {i: {"title": f"Movie {i}"} for i in range(int(item_emb.shape[0]))}

        mappings = {"item_to_movie": {i: i for i in range(int(item_emb.shape[0]))}}

    #user selection
    num_users = user_emb.shape[0]
    if user_id < 0 or user_id >= num_users:
        print(f"Warning: user_id {user_id} out of range, using 0 instead")
        user_id = 0

    scores = (
        np.dot(item_emb, user_emb[user_id])
        + item_bias
        + user_bias[user_id]
    )

    top_items = np.argsort(scores)[-top_k:][::-1]

    #prints out the user recommendations
    print(f"\nTop-{top_k} Recommendations for User {user_id}:\n")
    for idx in top_items:
        movie_id = mappings["item_to_movie"][int(idx)]
        # metadata may be a DataFrame or dict-like
        try:
            title = metadata.loc[movie_id]["title"]
        except Exception:
            try:
                title = metadata[movie_id]["title"]
            except Exception:
                title = f"Movie {movie_id}"
        print(f" → {title}  (score: {scores[int(idx)]:.3f})")

    print()


if __name__ == "__main__":
    recommend_for_user(user_id=123, top_k=10)
