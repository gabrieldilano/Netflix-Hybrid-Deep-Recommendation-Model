#!/usr/bin/env python3
"""
Netflix-style movie rating onboarding script (Hybrid Edition).

Merges Collaborative Filtering (Behavior) and Content-Based Filtering (Semantics)
into a weighted hybrid ensemble for recommendations.
"""

import pickle
import shutil
import urllib.request
import zipfile
import pandas as pd
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- Configuration ---
SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# CF Model Paths
MODEL_DIR = PROJECT_ROOT / "code" / "saved_model_cf"
# CBF Model Paths
CBF_DIR = PROJECT_ROOT / "code" / "saved_model_cbf"
USER_DIR = Path(__file__).resolve().parent / "user_embeddings"

EMBEDDING_DIM = 64
MIN_RATINGS = 15
MAX_RATINGS = 20

# Hybrid Hyperparameter
# 0.0 = Pure Content-Based (User Metadata Profile)
# 1.0 = Pure Collaborative (User Interaction History)
# 0.8 is a common starting point: rely mostly on behavior, use content to refine/break ties.
HYBRID_ALPHA = 0.8 

def download_movielens(data_dir: Path) -> None:
    """Download and extract MovieLens 20M dataset, including tags."""
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    zip_path = data_dir / "ml-20m.zip"

    required_files = ["movies.csv", "ratings.csv", "tags.csv"]
    if all((data_dir / f).exists() for f in required_files):
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading MovieLens 20M dataset (~200MB)...")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
    print("\nExtracting...")
    
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("ml-20m/movies.csv", data_dir)
        z.extract("ml-20m/ratings.csv", data_dir)
        z.extract("ml-20m/tags.csv", data_dir)

    shutil.move(str(data_dir / "ml-20m/movies.csv"), str(data_dir / "movies.csv"))
    shutil.move(str(data_dir / "ml-20m/ratings.csv"), str(data_dir / "ratings.csv"))
    shutil.move(str(data_dir / "ml-20m/tags.csv"), str(data_dir / "tags.csv"))
    
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
    item_embeddings = np.load(model_dir / "item_embeddings.npy")
    num_items = item_embeddings.shape[0]

    movies_path = data_dir / "movies.csv"
    movie_to_idx = {}
    idx_to_movie = {}
    movie_titles = {}
    movie_genres = {}
    genre_to_movies = defaultdict(list)

    with open(movies_path, encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            if idx >= num_items: break
            
            # Simple CSV parsing
            parts = line.strip().split(",", 2)
            if len(parts) < 3: continue
            
            movie_id = int(parts[0])
            rest = parts[1] + "," + parts[2] if len(parts) > 2 else parts[1]
            
            if rest.startswith('"'):
                end_quote = rest.find('",', 1)
                title = rest[1:end_quote]
                genres_str = rest[end_quote + 2 :]
            else:
                comma_idx = rest.rfind(",")
                title = rest[:comma_idx]
                genres_str = rest[comma_idx + 1 :]

            movie_to_idx[movie_id] = idx
            idx_to_movie[idx] = movie_id
            movie_titles[idx] = title
            genres = [g.strip() for g in genres_str.split("|") if g.strip()]
            movie_genres[idx] = genres

            for genre in genres:
                if genre and genre != "(no genres listed)":
                    genre_to_movies[genre].append(idx)

    # Compute popularity/variance
    ratings_path = data_dir / "ratings.csv"
    movie_ratings = defaultdict(list)
    if ratings_path.exists():
        with open(ratings_path, encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    mid = int(parts[1])
                    if mid in movie_to_idx:
                        movie_ratings[movie_to_idx[mid]].append(float(parts[2]))

    movie_popularity = {}
    movie_variance = {}
    movie_avg_rating = {}
    for idx in range(num_items):
        r = movie_ratings.get(idx, [])
        movie_popularity[idx] = len(r)
        if len(r) >= 2:
            movie_variance[idx] = float(np.var(r))
            movie_avg_rating[idx] = float(np.mean(r))
        else:
            movie_variance[idx] = 0.0
            movie_avg_rating[idx] = 0.0

    # Get top 50 most popular movies (by number of ratings)
    # Using 50 to ensure users have enough movies to choose from even if they skip many
    top_50_movies = sorted(
        range(num_items), 
        key=lambda idx: movie_popularity[idx], 
        reverse=True
    )[:50]

    mappings = {
        "movie_to_idx": movie_to_idx,
        "idx_to_movie": idx_to_movie,
        "movie_titles": movie_titles,
        "movie_genres": movie_genres,
        "genre_to_movies": dict(genre_to_movies),
        "movie_popularity": movie_popularity,
        "movie_variance": movie_variance,
        "movie_avg_rating": movie_avg_rating,
        "top_50_movies": top_50_movies,
        "num_items": num_items,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(mappings_path, "wb") as f:
        pickle.dump(mappings, f)
    return mappings

class AdaptiveSelector:
    """Selects movies to maximize information gain across genres."""
    def __init__(self, genre_to_movies, movie_genres, movie_popularity, movie_variance, movie_titles, min_popularity=100):
        self.genre_to_movies = genre_to_movies
        self.movie_genres = movie_genres
        self.movie_popularity = movie_popularity
        self.movie_variance = movie_variance
        self.movie_titles = movie_titles
        self.min_popularity = min_popularity
        self.genre_coverage = defaultdict(int)
        self.rated_indices = set()
        self.eligible_movies = {idx for idx, pop in movie_popularity.items() if pop >= min_popularity and movie_titles.get(idx)}
        self.major_genres = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Horror", "Animation"]

    def next_movie(self):
        available_genres = [g for g in self.major_genres if g in self.genre_to_movies and any(m in self.eligible_movies and m not in self.rated_indices for m in self.genre_to_movies[g])]
        if not available_genres:
            available_genres = [g for g in self.genre_to_movies if any(m in self.eligible_movies and m not in self.rated_indices for m in self.genre_to_movies[g])]
        
        if not available_genres: return None

        target_genre = min(available_genres, key=lambda g: self.genre_coverage[g])
        candidates = [m for m in self.genre_to_movies[target_genre] if m in self.eligible_movies and m not in self.rated_indices]
        
        if not candidates: return None

        # Score by Variance * 0.7 + Popularity * 0.3
        def score(m):
            return self.movie_variance.get(m, 0) * 0.7 + min(self.movie_popularity.get(m, 0)/10000, 1.0) * 0.3
        
        return max(candidates, key=score)

    def record_rating(self, movie_idx):
        self.rated_indices.add(movie_idx)
        for genre in self.movie_genres.get(movie_idx, []):
            self.genre_coverage[genre] += 1

def fit_user_embedding(rated_items, ratings, item_embeddings, item_biases, embedding_dim=64, lr=0.05, epochs=200, reg=1e-4):
    """Fit user embedding via gradient descent (Collaborative Filtering)."""
    weights = np.array(ratings) / sum(ratings)
    user_emb = np.zeros(embedding_dim)
    for idx, w in zip(rated_items, weights):
        user_emb += w * item_embeddings[idx]
    user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8) * 0.5
    user_bias = float(np.mean(ratings)) - 3.5

    rated_items_arr = np.array(rated_items)
    ratings_arr = np.array(ratings)
    best_loss = float("inf")
    patience, patience_counter = 20, 0

    for epoch in range(epochs):
        total_loss = 0.0
        order = np.random.permutation(len(rated_items_arr))
        for i in order:
            item_idx = rated_items_arr[i]
            actual = ratings_arr[i]
            item_emb = item_embeddings[item_idx]
            item_b = item_biases[item_idx]
            pred = np.dot(user_emb, item_emb) + user_bias + item_b
            error = pred - actual
            total_loss += error**2
            user_emb -= lr * (error * item_emb + reg * user_emb)
            user_bias -= lr * error

        total_loss /= len(rated_items_arr)
        if total_loss < best_loss - 1e-6:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break
        if epoch > 0 and epoch % 50 == 0: lr *= 0.5
    
    return user_emb, user_bias

# --- NEW: Hybrid Logic ---

def get_user_content_profile(rated_movies: dict, content_embeddings: np.ndarray) -> np.ndarray:
    """
    Creates a 'User Content Profile' by averaging the Sentence Transformer embeddings
    of movies the user liked (Rating >= 3.5).
    """
    liked_indices = [idx for idx, rating in rated_movies.items() if rating >= 3.5]
    
    # If no movies liked, return zero vector (neutral)
    if not liked_indices:
        return np.zeros(content_embeddings.shape[1])
    
    # Retrieve embeddings for liked movies
    liked_vectors = content_embeddings[liked_indices]
    
    # Average them to create a "Taste Vector"
    user_content_profile = np.mean(liked_vectors, axis=0)
    
    # Normalize
    norm = np.linalg.norm(user_content_profile)
    if norm > 0:
        user_content_profile /= norm
        
    return user_content_profile

def hybrid_recommend(
    user_cf_emb: np.ndarray,
    user_bias: float,
    user_content_profile: np.ndarray, 
    item_cf_embeddings: np.ndarray,
    item_biases: np.ndarray,
    content_embeddings: np.ndarray,  
    rated_indices: set[int],
    movie_titles: dict,
    movie_genres: dict,
    top_k: int = 10,
    alpha: float = 0.8 # New Parameter
):
    """
    Generates recommendations using a weighted ensemble of CF and CBF.
    Score = alpha * CF_Score + (1 - alpha) * CBF_Score
    """
    # 1. Calculate CF Scores (Dot Product)
    # Range: Roughly 1.0 to 5.0 (Predicted Rating)
    cf_raw_scores = np.dot(item_cf_embeddings, user_cf_emb) + item_biases + user_bias
    
    # Normalize CF scores to 0-1 range for fair combination
    # We assume reasonable bounds [1.0, 5.0] for clipping
    cf_norm_scores = (np.clip(cf_raw_scores, 1.0, 5.0) - 1.0) / 4.0
    
    # 2. Calculate CBF Scores (Cosine Similarity)
    # Range: -1.0 to 1.0 (but usually 0.0 to 1.0 for semantic text)
    if np.all(user_content_profile == 0):
        # User hasn't liked anything enough to form a profile
        cbf_scores = np.zeros_like(cf_norm_scores)
    else:
        # Dot product of normalized vectors == Cosine Similarity
        cbf_scores = np.dot(content_embeddings, user_content_profile)
        # Clip negative similarities (rare in SBERT but possible) to 0
        cbf_scores = np.clip(cbf_scores, 0.0, 1.0)

    # 3. Hybrid Merge
    final_scores = (alpha * cf_norm_scores) + ((1 - alpha) * cbf_scores)

    # 4. Mask rated movies
    for idx in rated_indices:
        final_scores[idx] = -np.inf

    # 5. Retrieve Top K
    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        title = movie_titles.get(idx, f"Movie {idx}")
        genres = movie_genres.get(idx, [])
        score = float(final_scores[idx])
        results.append((idx, title, genres, score))

    return results


def post_process_recommendations(
    candidate_results: list[tuple[int, str, list[str], float]],
    mappings: dict,
    top_k: int = 10,
    trust_weight: float = 0.7,
    diversity_weight: float = 0.2,
    novelty_weight: float = 0.1,
) -> list[tuple[int, str, list[str], float, dict]]:
    """
    Netflix-style post-processing: Prioritize high-confidence matches while maintaining
    diversity and including some novelty. No strict quotas - uses weighted scoring.
    
    Netflix's approach:
    - Relevance (trust/confidence) is primary - users want accurate predictions
    - Diversity prevents redundancy (don't show 10 action movies)
    - Novelty adds serendipity but is secondary to relevance
    
    Args:
        candidate_results: List of (idx, title, genres, score) tuples
        mappings: Dictionary with movie_popularity, movie_variance, movie_avg_rating
        top_k: Number of final recommendations
        trust_weight: Weight for confidence/trust (default 0.7 - high priority)
        diversity_weight: Weight for genre diversity (default 0.2 - moderate)
        novelty_weight: Weight for less-known movies (default 0.1 - low priority)
    
    Returns:
        List of (idx, title, genres, final_score, metadata) tuples
        where metadata contains trust, diversity, novelty scores
    """
    if not candidate_results:
        return []
    
    movie_popularity = mappings.get("movie_popularity", {})
    movie_variance = mappings.get("movie_variance", {})
    movie_avg_rating = mappings.get("movie_avg_rating", {})
    
    # Normalize weights
    total_weight = trust_weight + diversity_weight + novelty_weight
    if total_weight > 0:
        trust_weight /= total_weight
        diversity_weight /= total_weight
        novelty_weight /= total_weight
    
    # Calculate metrics for each candidate
    candidates_with_metrics = []
    for idx, title, genres, base_score in candidate_results:
        popularity = movie_popularity.get(idx, 0)
        avg_rating = movie_avg_rating.get(idx, 0)
        
        # Trust/Confidence: High prediction score + data reliability
        # More ratings = more confidence in the prediction
        max_popularity = max(movie_popularity.values()) if movie_popularity else 1
        # Trust combines prediction quality (base_score) with data reliability
        trust_score = base_score * 0.8 + (popularity / max_popularity) * 0.2
        
        # Novelty: Less popular movies (inverse of popularity)
        # Normalize to 0-1 range
        novelty_score = 1.0 - min(popularity / max_popularity, 1.0) if max_popularity > 0 else 0.5
        
        candidates_with_metrics.append({
            'idx': idx,
            'title': title,
            'genres': set(genres) if genres else set(),
            'base_score': base_score,
            'trust_score': trust_score,
            'novelty_score': novelty_score,
            'popularity': popularity,
            'avg_rating': avg_rating,
        })
    
    # Netflix-style greedy selection: balance relevance with diversity
    selected = []
    selected_genres = set()
    
    # Sort candidates by initial relevance (trust score)
    candidates_with_metrics.sort(key=lambda x: x['trust_score'], reverse=True)
    
    # Greedy selection: pick items that maximize weighted score
    # This naturally favors high-confidence items while maintaining diversity
    while len(selected) < top_k and candidates_with_metrics:
        best_candidate = None
        best_combined_score = -float('inf')
        
        for candidate in candidates_with_metrics:
            # Calculate diversity: how different is this from already selected?
            genre_overlap = len(candidate['genres'] & selected_genres)
            total_genres = len(candidate['genres']) if candidate['genres'] else 1
            diversity_score = 1.0 - (genre_overlap / max(total_genres, 1))
            
            # Netflix-style combined score: heavily weighted toward trust
            # Diversity and novelty provide boosts but don't override relevance
            combined_score = (
                trust_weight * candidate['trust_score'] +
                diversity_weight * diversity_score +
                novelty_weight * candidate['novelty_score']
            )
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
            selected_genres.update(best_candidate['genres'])
            candidates_with_metrics.remove(best_candidate)
    
    # Format results with metadata
    results = []
    accumulated_genres = set()
    
    for candidate in selected:
        # Calculate diversity relative to previously selected items
        genre_overlap = len(candidate['genres'] & accumulated_genres)
        total_genres = len(candidate['genres']) if candidate['genres'] else 1
        diversity_score = 1.0 - (genre_overlap / max(total_genres, 1))
        
        # Final combined score for display
        final_score = (
            trust_weight * candidate['trust_score'] +
            diversity_weight * diversity_score +
            novelty_weight * candidate['novelty_score']
        )
        
        # Determine category for display
        category = 'trust' if candidate['trust_score'] >= 0.85 else \
                   'novelty' if candidate['novelty_score'] > 0.6 else \
                   'diversity' if diversity_score > 0.5 else 'standard'
        
        metadata = {
            'trust': candidate['trust_score'],
            'novelty': candidate['novelty_score'],
            'diversity': diversity_score,
            'popularity': candidate['popularity'],
            'avg_rating': candidate['avg_rating'],
            'category': category
        }
        
        results.append((
            candidate['idx'],
            candidate['title'],
            list(candidate['genres']),
            final_score,
            metadata
        ))
        
        accumulated_genres.update(candidate['genres'])
    
    return results

# --- Main Flow ---

def save_user_profile(username, user_embedding, user_bias, rated_movies, user_dir):
    user_dir.mkdir(parents=True, exist_ok=True)
    filepath = user_dir / f"{username}.npz"
    np.savez(filepath, embedding=user_embedding, bias=np.array([user_bias]), rated_indices=np.array(list(rated_movies.keys())), rated_scores=np.array(list(rated_movies.values())))
    return filepath

def load_user_profile(username, user_dir):
    filepath = user_dir / f"{username}.npz"
    if not filepath.exists(): return None
    data = np.load(filepath)
    return (data["embedding"], float(data["bias"][0]), dict(zip(data["rated_indices"].tolist(), data["rated_scores"].tolist())))

def parse_rating(input_str):
    input_str = input_str.strip().lower()
    if input_str in ("s", "skip", ""): return None
    try:
        r = float(input_str)
        r = round(r * 2) / 2
        return r if 0.5 <= r <= 5.0 else -1
    except ValueError: return -1

def main():
    print("\n" + "=" * 60)
    print("    CineMatch")
    print("=" * 60 + "\n")

    download_movielens(DATA_DIR)

    print("Loading Collaborative Filtering model...")
    item_cf_embeddings = np.load(MODEL_DIR / "item_embeddings.npy")
    item_biases = np.load(MODEL_DIR / "item_biases.npy")
    
    print("Loading Content-Based model (Sentence Transformers)...")
    content_path = CBF_DIR / "content_embeddings.npy"
    if content_path.exists():
        raw_content_embeddings = np.load(content_path) # Shape: (27278, 384)
    else:
        print("WARNING: Content embeddings not found. Run cbf_model.ipynb first!")
        raw_content_embeddings = np.zeros((item_cf_embeddings.shape[0], 384))

    mappings = load_or_generate_mappings(DATA_DIR, MODEL_DIR)

    # The CF model (item_cf_embeddings) uses a specific subset of movies (26,744).
    # The CBF model (raw_content_embeddings) uses all movies in movies.csv (27,278).
    # We must slice and reorder the CBF embeddings to match the CF rows.
    
    if raw_content_embeddings.shape[0] != item_cf_embeddings.shape[0]:
        print(f"Aligning matrices: CF {item_cf_embeddings.shape} vs CBF {raw_content_embeddings.shape}...")
        
        # 1. Load movies.csv to know which row in 'raw_content_embeddings' belongs to which movieId
        # (CBF embeddings were generated sequentially from movies.csv)
        movies_df = pd.read_csv(DATA_DIR / "movies.csv")
        # Map Real MovieID -> Row Index in raw_content_embeddings
        mid_to_cbf_idx = {mid: i for i, mid in enumerate(movies_df['movieId'])}
        
        # 2. Create a new aligned matrix matching CF shape
        aligned_content = np.zeros((item_cf_embeddings.shape[0], raw_content_embeddings.shape[1]))
        
        # 3. Fill it using the CF mapping (idx_to_movie)
        aligned_count = 0
        for cf_idx in range(item_cf_embeddings.shape[0]):
            real_movie_id = mappings["idx_to_movie"][cf_idx]
            
            if real_movie_id in mid_to_cbf_idx:
                cbf_idx = mid_to_cbf_idx[real_movie_id]
                aligned_content[cf_idx] = raw_content_embeddings[cbf_idx]
                aligned_count += 1
            else:
                # This movie exists in CF but not in movies.csv (very rare/impossible if data is clean)
                pass
                
        print(f"Alignment complete. Matched {aligned_count} movies.")
        content_embeddings = aligned_content
    else:
        content_embeddings = raw_content_embeddings

    username = input("Enter your username: ").strip() or "default_user"
    existing = load_user_profile(username, USER_DIR)
    
    rated_movies = {}
    user_cf_emb = None
    user_bias = 0.0

    if existing:
        user_cf_emb, user_bias, rated_movies = existing
        print(f"\nWelcome back, {username}! ({len(rated_movies)} rated movies)")
        if input("Continue with existing profile? (y/n): ").strip().lower() == "y":
            
            # Generate Profile on the fly
            user_content_profile = get_user_content_profile(rated_movies, content_embeddings)
            
            recs = hybrid_recommend(
                user_cf_emb, user_bias, user_content_profile,
                item_cf_embeddings, item_biases, content_embeddings,
                set(rated_movies.keys()),
                mappings["movie_titles"], mappings["movie_genres"],
                alpha=HYBRID_ALPHA
            )
            
            print("\n" + "-" * 50)
            print("Your Recommendations:")
            print("-" * 50)
            for i, (idx, title, genres, score) in enumerate(recs, 1):
                print(f"{i:2}. {title} [{'|'.join(genres[:3])}] (Score: {score:.3f})")
            return
        else:
            rated_movies = {}

    print(f"\nHi {username}! Let's build your taste profile.")
    print(f"Rate movies from the most popular (1-5 stars, or 's' to skip).")
    print(f"You need at least {MIN_RATINGS} ratings to get recommendations.\n")
    
    # Use top 50 most popular movies (more than needed so users can skip some)
    top_50 = mappings["top_50_movies"]
    
    # Filter out already-rated movies
    available_movies = [idx for idx in top_50 if idx not in rated_movies]
    
    ratings_collected = len(rated_movies)
    movie_position = 0

    while ratings_collected < MAX_RATINGS and movie_position < len(available_movies):
        movie_idx = available_movies[movie_position]
        movie_position += 1

        title = mappings["movie_titles"].get(movie_idx, f"Movie {movie_idx}")
        genres = mappings["movie_genres"].get(movie_idx, [])
        genre_str = "|".join(genres[:3]) if genres else "Unknown"
        num_ratings = mappings["movie_popularity"].get(movie_idx, 0)

        # Show progress toward MIN_RATINGS (15)
        progress_indicator = f"[{ratings_collected}/{MIN_RATINGS}]" if ratings_collected < MIN_RATINGS else f"[{ratings_collected}+]"
        print(f"{progress_indicator} {title} [{genre_str}] ({num_ratings:,} ratings)")
        
        rating = -1
        while rating == -1:
            rating = parse_rating(input("Rating (1-5, s=skip): "))

        if rating is not None:
            rated_movies[movie_idx] = rating
            ratings_collected += 1
            
            # Show encouragement message after 15 ratings
            if ratings_collected >= MIN_RATINGS:
                print(f"✓ You've rated {ratings_collected} movies! Feel free to keep rating for better recommendations.")
        # else: skip - just continue to next movie

        # Allow continuing after MIN_RATINGS, but don't force prompt every time
        if ratings_collected >= MIN_RATINGS and ratings_collected < MAX_RATINGS:
            if ratings_collected == MIN_RATINGS:
                # Only ask once when they hit 15
                if input(f"\n{ratings_collected} ratings collected. Continue rating? (y/n): ").strip().lower() != "y":
                    break
                print()
            elif ratings_collected >= MAX_RATINGS:
                break

    print("\nFitting your taste profile...", end=" ", flush=True)
    
    # 1. Fit CF
    user_cf_emb, user_bias = fit_user_embedding(
        list(rated_movies.keys()), list(rated_movies.values()),
        item_cf_embeddings, item_biases, embedding_dim=EMBEDDING_DIM
    )
    
    # 2. Fit Content Profile
    user_content_profile = get_user_content_profile(rated_movies, content_embeddings)
    
    print("done!")

    # Get candidate recommendations
    candidate_recs = hybrid_recommend(
        user_cf_emb, user_bias, user_content_profile,
        item_cf_embeddings, item_biases, content_embeddings,
        set(rated_movies.keys()),
        mappings["movie_titles"], mappings["movie_genres"],
        alpha=HYBRID_ALPHA,
        top_k=50  # Get more candidates for post-processing
    )
    
    # Post-process with Netflix-style weighted scoring
    recs = post_process_recommendations(
        candidate_recs,
        mappings,
        top_k=10,
        trust_weight=0.7,  # High priority: relevance and confidence
        diversity_weight=0.2,  # Moderate: prevent redundancy
        novelty_weight=0.1,  # Low priority: add serendipity
    )

    print("\n" + "-" * 50)
    print("Your Recommendations:")
    print("-" * 50)
    for i, rec in enumerate(recs, 1):
        if len(rec) == 5:
            idx, title, genres, score, metadata = rec
            category = metadata.get('category', '')
            badge = ""
            if category == 'trust':
                badge = " 🛡️ High Confidence"
            elif category == 'novelty':
                badge = " ✨ Hidden Gem"
            elif category == 'diversity':
                badge = " 🎨 Diverse"
            print(f"{i:2}. {title} [{'|'.join(genres[:3])}] (Score: {score:.3f}){badge}")
        else:
            # Fallback for old format
            idx, title, genres, score = rec
            print(f"{i:2}. {title} [{'|'.join(genres[:3])}] (Score: {score:.3f})")

    save_user_profile(username, user_cf_emb, user_bias, rated_movies, USER_DIR)

if __name__ == "__main__":
    main()