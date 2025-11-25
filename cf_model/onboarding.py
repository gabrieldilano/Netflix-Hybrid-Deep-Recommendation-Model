#!/usr/bin/env python3
"""
Netflix-style movie rating onboarding script.

Lets new users rate 15-20 movies, fits a personalized user embedding via
gradient descent, and generates movie recommendations.
"""

import pickle
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
USER_DIR = Path(__file__).resolve().parent / "user_embeddings"

EMBEDDING_DIM = 64
MIN_RATINGS = 15
MAX_RATINGS = 20


def download_movielens(data_dir: Path) -> None:
    """Download and extract MovieLens 20M dataset."""
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    zip_path = data_dir / "ml-20m.zip"

    if (data_dir / "movies.csv").exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MovieLens 20M dataset (~200MB)...")
    print("This may take a few minutes...")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
    print("\n")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("ml-20m/movies.csv", data_dir)
        z.extract("ml-20m/ratings.csv", data_dir)

    # Move to expected locations
    shutil.move(str(data_dir / "ml-20m/movies.csv"), str(data_dir / "movies.csv"))
    shutil.move(str(data_dir / "ml-20m/ratings.csv"), str(data_dir / "ratings.csv"))
    zip_path.unlink()
    shutil.rmtree(data_dir / "ml-20m")

    print("Download complete!")


def load_or_generate_mappings(data_dir: Path, model_dir: Path) -> dict:
    """Load mappings from cache or generate from movie/rating data."""
    mappings_path = model_dir / "mappings.pkl"

    if mappings_path.exists():
        print("Loading cached mappings...")
        with open(mappings_path, "rb") as f:
            return pickle.load(f)

    print("Generating mappings from data (first run only)...")

    # Load item embeddings to get the exact movie indices used in training
    item_embeddings = np.load(model_dir / "item_embeddings.npy")
    num_items = item_embeddings.shape[0]

    # Load movies.csv
    movies_path = data_dir / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"movies.csv not found at {movies_path}")

    movie_to_idx = {}
    idx_to_movie = {}
    movie_titles = {}
    movie_genres = {}
    genre_to_movies = defaultdict(list)

    with open(movies_path, encoding="utf-8") as f:
        f.readline()  # Skip header
        for idx, line in enumerate(f):
            if idx >= num_items:
                break

            parts = line.strip().split(",", 2)
            if len(parts) < 3:
                continue

            movie_id = int(parts[0])
            # Handle titles with commas (they're quoted)
            rest = parts[1] + "," + parts[2] if len(parts) > 2 else parts[1]
            if rest.startswith('"'):
                # Find the closing quote
                end_quote = rest.find('",', 1)
                if end_quote != -1:
                    title = rest[1:end_quote]
                    genres_str = rest[end_quote + 2 :]
                else:
                    title = rest[1:-1] if rest.endswith('"') else rest[1:]
                    genres_str = ""
            else:
                comma_idx = rest.rfind(",")
                if comma_idx != -1:
                    title = rest[:comma_idx]
                    genres_str = rest[comma_idx + 1 :]
                else:
                    title = rest
                    genres_str = ""

            movie_to_idx[movie_id] = idx
            idx_to_movie[idx] = movie_id
            movie_titles[idx] = title
            genres = [g.strip() for g in genres_str.split("|") if g.strip()]
            movie_genres[idx] = genres

            for genre in genres:
                if genre and genre != "(no genres listed)":
                    genre_to_movies[genre].append(idx)

    # Compute popularity and variance from ratings
    print("Computing movie statistics from ratings...")
    ratings_path = data_dir / "ratings.csv"

    movie_ratings = defaultdict(list)
    if ratings_path.exists():
        with open(ratings_path, encoding="utf-8") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    movie_id = int(parts[1])
                    if movie_id in movie_to_idx:
                        rating = float(parts[2])
                        movie_ratings[movie_to_idx[movie_id]].append(rating)

    movie_popularity = {}
    movie_variance = {}
    for idx in range(num_items):
        ratings = movie_ratings.get(idx, [])
        movie_popularity[idx] = len(ratings)
        if len(ratings) >= 2:
            movie_variance[idx] = float(np.var(ratings))
        else:
            movie_variance[idx] = 0.0

    mappings = {
        "movie_to_idx": movie_to_idx,
        "idx_to_movie": idx_to_movie,
        "movie_titles": movie_titles,
        "movie_genres": movie_genres,
        "genre_to_movies": dict(genre_to_movies),
        "movie_popularity": movie_popularity,
        "movie_variance": movie_variance,
        "num_items": num_items,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(mappings_path, "wb") as f:
        pickle.dump(mappings, f)

    print(f"Saved mappings to {mappings_path}")
    return mappings


class AdaptiveSelector:
    """Selects movies to maximize information gain across genres."""

    def __init__(
        self,
        genre_to_movies: dict[str, list[int]],
        movie_genres: dict[int, list[str]],
        movie_popularity: dict[int, int],
        movie_variance: dict[int, float],
        movie_titles: dict[int, str],
        min_popularity: int = 100,
    ):
        self.genre_to_movies = genre_to_movies
        self.movie_genres = movie_genres
        self.movie_popularity = movie_popularity
        self.movie_variance = movie_variance
        self.movie_titles = movie_titles
        self.min_popularity = min_popularity

        self.genre_coverage = defaultdict(int)
        self.rated_indices: set[int] = set()

        # Filter to well-known movies only
        self.eligible_movies = {
            idx
            for idx, pop in movie_popularity.items()
            if pop >= min_popularity and movie_titles.get(idx)
        }

        # Major genres to ensure coverage
        self.major_genres = [
            "Action",
            "Comedy",
            "Drama",
            "Thriller",
            "Romance",
            "Sci-Fi",
            "Horror",
            "Animation",
            "Documentary",
            "Adventure",
        ]

    def next_movie(self) -> int | None:
        """Select next movie prioritizing underrepresented genres."""
        # Find genre with least coverage that has eligible movies
        available_genres = [
            g
            for g in self.major_genres
            if g in self.genre_to_movies
            and any(
                m in self.eligible_movies and m not in self.rated_indices
                for m in self.genre_to_movies[g]
            )
        ]

        if not available_genres:
            # Fall back to any genre
            available_genres = [
                g
                for g in self.genre_to_movies
                if any(
                    m in self.eligible_movies and m not in self.rated_indices
                    for m in self.genre_to_movies[g]
                )
            ]

        if not available_genres:
            return None

        # Pick least-covered genre
        target_genre = min(available_genres, key=lambda g: self.genre_coverage[g])

        # Get candidates from that genre
        candidates = [
            m
            for m in self.genre_to_movies[target_genre]
            if m in self.eligible_movies and m not in self.rated_indices
        ]

        if not candidates:
            return None

        # Score by variance (controversial = more signal) and popularity
        def score(m):
            var_score = self.movie_variance.get(m, 0)
            pop_score = min(self.movie_popularity.get(m, 0) / 10000, 1.0)
            return var_score * 0.7 + pop_score * 0.3

        return max(candidates, key=score)

    def record_rating(self, movie_idx: int) -> None:
        """Record that a movie was rated."""
        self.rated_indices.add(movie_idx)
        for genre in self.movie_genres.get(movie_idx, []):
            self.genre_coverage[genre] += 1


def fit_user_embedding(
    rated_items: list[int],
    ratings: list[float],
    item_embeddings: np.ndarray,
    item_biases: np.ndarray,
    embedding_dim: int = 64,
    lr: float = 0.05,
    epochs: int = 200,
    reg: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """
    Fit user embedding via gradient descent to minimize MSE on seed ratings.

    Returns (user_embedding, user_bias).
    """
    # Initialize with weighted average of rated item embeddings
    weights = np.array(ratings) / sum(ratings)
    user_emb = np.zeros(embedding_dim)
    for idx, w in zip(rated_items, weights):
        user_emb += w * item_embeddings[idx]
    user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8) * 0.5

    # Initialize bias near mean rating
    user_bias = float(np.mean(ratings)) - 3.5

    rated_items_arr = np.array(rated_items)
    ratings_arr = np.array(ratings)

    best_loss = float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0.0

        # Shuffle order each epoch
        order = np.random.permutation(len(rated_items_arr))

        for i in order:
            item_idx = rated_items_arr[i]
            actual = ratings_arr[i]

            item_emb = item_embeddings[item_idx]
            item_b = item_biases[item_idx]

            pred = np.dot(user_emb, item_emb) + user_bias + item_b
            error = pred - actual
            total_loss += error**2

            # Gradient updates
            user_emb -= lr * (error * item_emb + reg * user_emb)
            user_bias -= lr * error

        total_loss /= len(rated_items_arr)

        # Early stopping
        if total_loss < best_loss - 1e-6:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # Decay learning rate
        if epoch > 0 and epoch % 50 == 0:
            lr *= 0.5

    return user_emb, user_bias


def recommend(
    user_embedding: np.ndarray,
    user_bias: float,
    item_embeddings: np.ndarray,
    item_biases: np.ndarray,
    rated_indices: set[int],
    movie_titles: dict[int, str],
    movie_genres: dict[int, list[str]],
    top_k: int = 10,
) -> list[tuple[int, str, list[str], float]]:
    """Generate top-k recommendations for a user."""
    # Score all items
    scores = np.dot(item_embeddings, user_embedding) + item_biases + user_bias

    # Mask rated movies
    for idx in rated_indices:
        scores[idx] = -np.inf

    # Top-k
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        title = movie_titles.get(idx, f"Movie {idx}")
        genres = movie_genres.get(idx, [])
        score = float(scores[idx])
        results.append((idx, title, genres, score))

    return results


def save_user_profile(
    username: str,
    user_embedding: np.ndarray,
    user_bias: float,
    rated_movies: dict[int, float],
    user_dir: Path,
) -> Path:
    """Save user profile to disk."""
    user_dir.mkdir(parents=True, exist_ok=True)
    filepath = user_dir / f"{username}.npz"

    np.savez(
        filepath,
        embedding=user_embedding,
        bias=np.array([user_bias]),
        rated_indices=np.array(list(rated_movies.keys())),
        rated_scores=np.array(list(rated_movies.values())),
    )

    return filepath


def load_user_profile(username: str, user_dir: Path) -> tuple | None:
    """Load existing user profile if it exists."""
    filepath = user_dir / f"{username}.npz"
    if not filepath.exists():
        return None

    data = np.load(filepath)
    return (
        data["embedding"],
        float(data["bias"][0]),
        dict(zip(data["rated_indices"].tolist(), data["rated_scores"].tolist())),
    )


def parse_rating(input_str: str) -> float | None:
    """Parse user rating input. Returns None for skip."""
    input_str = input_str.strip().lower()

    if input_str in ("s", "skip", ""):
        return None

    try:
        rating = float(input_str)
        # Round to nearest 0.5
        rating = round(rating * 2) / 2
        if 0.5 <= rating <= 5.0:
            return rating
        print("Rating must be between 0.5 and 5.0")
        return -1  # Invalid but not skip
    except ValueError:
        print("Invalid input. Enter a number 1-5 or 's' to skip.")
        return -1


def main():
    print("\n" + "=" * 60)
    print("    Movie Recommendation Onboarding")
    print("=" * 60 + "\n")

    # Download data if needed
    download_movielens(DATA_DIR)

    # Load embeddings
    print("Loading model...")
    item_embeddings = np.load(MODEL_DIR / "item_embeddings.npy")
    item_biases = np.load(MODEL_DIR / "item_biases.npy")

    # Load or generate mappings
    mappings = load_or_generate_mappings(DATA_DIR, MODEL_DIR)

    # Get username
    username = input("Enter your username: ").strip()
    if not username:
        username = "default_user"

    # Check for existing profile
    existing = load_user_profile(username, USER_DIR)
    if existing:
        user_emb, user_bias, rated_movies = existing
        print(f"\nWelcome back, {username}!")
        print(f"You've rated {len(rated_movies)} movies previously.")

        choice = input("Continue with existing profile? (y/n): ").strip().lower()
        if choice == "y":
            # Show recommendations directly
            print("\n" + "-" * 40)
            print("Your Top 10 Recommendations:")
            print("-" * 40)

            recs = recommend(
                user_emb,
                user_bias,
                item_embeddings,
                item_biases,
                set(rated_movies.keys()),
                mappings["movie_titles"],
                mappings["movie_genres"],
            )

            for i, (idx, title, genres, score) in enumerate(recs, 1):
                genre_str = "|".join(genres[:3]) if genres else "Unknown"
                print(f"{i:2}. {title} [{genre_str}] - {score:.1f}*")

            return
        else:
            rated_movies = {}
    else:
        rated_movies = {}

    print(f"\nHi {username}! Let's build your taste profile.")
    print(f"Rate {MIN_RATINGS}-{MAX_RATINGS} movies (1-5 stars, or 's' to skip).\n")

    # Initialize selector
    selector = AdaptiveSelector(
        genre_to_movies=mappings["genre_to_movies"],
        movie_genres=mappings["movie_genres"],
        movie_popularity=mappings["movie_popularity"],
        movie_variance=mappings["movie_variance"],
        movie_titles=mappings["movie_titles"],
    )

    # Mark already-rated movies
    for idx in rated_movies:
        selector.record_rating(idx)

    ratings_collected = len(rated_movies)
    shown = 0

    while ratings_collected < MAX_RATINGS:
        movie_idx = selector.next_movie()
        if movie_idx is None:
            print("No more movies available to rate.")
            break

        shown += 1
        title = mappings["movie_titles"].get(movie_idx, f"Movie {movie_idx}")
        genres = mappings["movie_genres"].get(movie_idx, [])
        genre_str = "|".join(genres[:3]) if genres else "Unknown"

        print(f"[{ratings_collected + 1}/{MAX_RATINGS}] {title} [{genre_str}]")
        rating = -1
        while rating == -1:
            rating = parse_rating(input("Your rating (1-5, s=skip): "))

        if rating is not None:
            rated_movies[movie_idx] = rating
            ratings_collected += 1
            selector.record_rating(movie_idx)
        else:
            # Skip - mark as seen but don't add to ratings
            selector.rated_indices.add(movie_idx)

        # Check if minimum reached
        if ratings_collected >= MIN_RATINGS:
            choice = input(f"\n{ratings_collected} ratings collected. Continue? (y/n): ")
            if choice.strip().lower() != "y":
                break
            print()

    if ratings_collected < MIN_RATINGS:
        print(f"\nNeed at least {MIN_RATINGS} ratings. You have {ratings_collected}.")
        return

    # Fit user embedding
    print("\nFitting your taste profile...", end=" ", flush=True)

    rated_indices = list(rated_movies.keys())
    ratings_list = list(rated_movies.values())

    user_emb, user_bias = fit_user_embedding(
        rated_indices,
        ratings_list,
        item_embeddings,
        item_biases,
        embedding_dim=EMBEDDING_DIM,
    )
    print("done!")

    # Generate recommendations
    print("\n" + "-" * 40)
    print("Your Top 10 Recommendations:")
    print("-" * 40)

    recs = recommend(
        user_emb,
        user_bias,
        item_embeddings,
        item_biases,
        set(rated_indices),
        mappings["movie_titles"],
        mappings["movie_genres"],
    )

    for i, (idx, title, genres, score) in enumerate(recs, 1):
        genre_str = "|".join(genres[:3]) if genres else "Unknown"
        print(f"{i:2}. {title} [{genre_str}] - {score:.1f}*")

    # Save profile
    filepath = save_user_profile(
        username, user_emb, user_bias, rated_movies, USER_DIR
    )
    print(f"\nProfile saved to {filepath}")


if __name__ == "__main__":
    main()
