"""
Streamlit UI for Movie Recommendation Onboarding
Interactive card-based interface for rating movies and getting recommendations
Uses the same logic and functions as onboarding.py for consistency
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# Import functions from onboarding.py
from onboarding import (
    load_or_generate_mappings,
    fit_user_embedding,
    get_user_content_profile,
    hybrid_recommend,
    post_process_recommendations,
    save_user_profile,
    load_user_profile,
    DATA_DIR,
    MODEL_DIR,
    CBF_DIR,
    USER_DIR,
    EMBEDDING_DIM,
    MIN_RATINGS,
    MAX_RATINGS,
    HYBRID_ALPHA,
)


# Custom CSS for beautiful cards and animations
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .movie-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 20px auto;
        max-width: 500px;
        text-align: center;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .movie-title {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
        line-height: 1.3;
    }
    
    .movie-genre {
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 10px;
    }
    
    .movie-stats {
        font-size: 14px;
        color: #95a5a6;
        margin-bottom: 25px;
    }
    
    /* Progress styling */
    .progress-text {
        text-align: center;
        color: white;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    
    /* Recommendation card styling */
    .rec-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .rec-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .rec-number {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        line-height: 35px;
        text-align: center;
        font-weight: bold;
        margin-right: 15px;
    }
    
    .rec-title {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    
    .rec-genre {
        font-size: 14px;
        color: #7f8c8d;
    }
    
    .rec-score {
        font-size: 16px;
        color: #667eea;
        font-weight: bold;
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        color: white;
        padding: 20px;
        margin-bottom: 30px;
    }
    
    .app-title {
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-subtitle {
        font-size: 20px;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.username = ""
        st.session_state.stage = "welcome"  # welcome, rating, results
        st.session_state.rated_movies = {}
        st.session_state.current_movie_idx = 0
        st.session_state.recommendations = []


def render_welcome_screen():
    """Render the welcome screen."""
    st.markdown("""
    <div class="app-header">
        <div class="app-title">CineMatch</div>
        <div class="app-subtitle">Discover your next favorite movie</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="movie-card">
            <h2 style="color: #000;">Welcome! 👋</h2>
            <p style="font-size: 16px; color: #555; margin: 20px 0;">
                Rate some popular movies to get personalized recommendations tailored just for you.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use form to handle Enter key submission properly
        with st.form("username_form", clear_on_submit=False):
            username = st.text_input("Enter your username:", key="username_input", 
                                    placeholder="e.g., movie_lover")
            
            # Check for existing user
            if username.strip():
                user_file = USER_DIR / f"{username.strip()}.npz"
                if user_file.exists():
                    st.info(f"👋 Welcome back, {username.strip()}! We found your existing profile.")
            
            submitted = st.form_submit_button("🚀 Start Rating Movies", use_container_width=True, type="primary")
            
            if submitted:
                if username.strip():
                    st.session_state.username = username.strip()
                    
                    # Check if user has existing ratings
                    try:
                        profile = load_user_profile(username.strip(), USER_DIR)
                        if profile:
                            st.session_state.rated_movies = profile['rated_movies']
                            st.info(f"Loaded {len(profile['rated_movies'])} existing ratings!")
                    except:
                        pass
                    
                    st.session_state.stage = "rating"
                    st.rerun()
                else:
                    st.warning("Please enter a username to continue!")


def render_movie_card(movie_idx: int, mappings: dict, ratings_collected: int):
    """Render a single movie card for rating."""
    title = mappings["movie_titles"].get(movie_idx, f"Movie {movie_idx}")
    genres = mappings["movie_genres"].get(movie_idx, [])
    genre_str = " | ".join(genres[:3]) if genres else "Unknown"
    num_ratings = mappings["movie_popularity"].get(movie_idx, 0)
    avg_rating = mappings.get("movie_avg_rating", {}).get(movie_idx, 0)
    
    # Progress bar: 0 to 15 (MIN_RATINGS), stays full after 15
    progress = min(ratings_collected / MIN_RATINGS, 1.0)
    st.progress(progress)
    
    # Movie card
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">{title}</div>
        <div class="movie-genre">🎭 {genre_str}</div>
        <div class="movie-stats">⭐ {num_ratings:,} ratings{f" • {avg_rating:.2f}★ avg" if avg_rating > 0 else ""}</div>
    </div>
    """, unsafe_allow_html=True)


def render_rating_screen(mappings: dict):
    """Render the rating screen with movie cards."""
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Rate Movies</div>
        <div class="app-subtitle">Swipe through and rate what you know</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use top 50 most popular movies from mappings
    top_50 = mappings.get("top_50_movies", [])
    
    if not top_50:
        st.error("No movies available. Please check your data configuration.")
        st.stop()
    
    available_movies = [idx for idx in top_50 if idx not in st.session_state.rated_movies]
    
    if st.session_state.current_movie_idx >= len(available_movies):
        # Check if we have enough ratings
        if len(st.session_state.rated_movies) >= MIN_RATINGS:
            st.session_state.stage = "generating"
            st.rerun()
        else:
            st.error(f"Not enough ratings! You need at least {MIN_RATINGS} ratings. You have {len(st.session_state.rated_movies)}.")
            if st.button("🔄 Start Over"):
                st.session_state.clear()
                st.rerun()
            st.stop()
    
    movie_idx = available_movies[st.session_state.current_movie_idx]
    ratings_collected = len(st.session_state.rated_movies)
    
    # Render the movie card
    render_movie_card(movie_idx, mappings, ratings_collected)
    
    # Show message if 15+ ratings collected
    if ratings_collected >= MIN_RATINGS:
        st.success("✅ You've rated 15 movies! Feel free to keep rating for better recommendations.")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Skip", use_container_width=True, key="skip_btn"):
            st.session_state.current_movie_idx += 1
            st.rerun()
    
    with col2:
        # Always show slider and submit button
        rating = st.select_slider(
            "Your rating:",
            options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            value=3.0,
            format_func=lambda x: f"{x:.1f}★",
            key=f"rating_slider_{movie_idx}"
        )
        
        if st.button("✅ Submit Rating", use_container_width=True, type="primary", key="submit_btn"):
            st.session_state.rated_movies[movie_idx] = float(rating)
            st.session_state.current_movie_idx += 1
            
            # Check if we want to continue or finish
            if len(st.session_state.rated_movies) >= MAX_RATINGS or st.session_state.current_movie_idx >= len(available_movies):
                st.session_state.stage = "generating"
            st.rerun()
    
    with col3:
        if len(st.session_state.rated_movies) >= MIN_RATINGS:
            if st.button("✨ Get Recommendations", use_container_width=True, key="finish_btn"):
                st.session_state.stage = "generating"
                st.rerun()
    
    # Display rating count
    st.markdown(f"""
    <div style="text-align: center; color: white; margin-top: 30px; font-size: 18px;">
        <strong>{ratings_collected}</strong> movies rated
        {f" • <span style='color: #90EE90;'>Ready to get recommendations!</span>" if ratings_collected >= MIN_RATINGS else f" • Need {MIN_RATINGS - ratings_collected} more"}
    </div>
    """, unsafe_allow_html=True)


def render_generating_screen():
    """Render the generating recommendations screen."""
    st.markdown("""
    <div class="app-header">
        <div class="app-title">✨ Generating Recommendations</div>
        <div class="app-subtitle">Analyzing your taste profile...</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🎬 Finding perfect movies for you..."):
        # Load models (same as onboarding.py)
        item_embeddings = np.load(MODEL_DIR / "item_embeddings.npy")
        item_biases = np.load(MODEL_DIR / "item_biases.npy")
        raw_content_embeddings = np.load(CBF_DIR / "content_embeddings.npy")
        mappings = st.session_state.mappings
        
        # Align content embeddings to match CF model (same logic as onboarding.py)
        if raw_content_embeddings.shape[0] != item_embeddings.shape[0]:
            # Load movies.csv to map movieId -> CBF row index
            movies_df = pd.read_csv(DATA_DIR / "movies.csv")
            mid_to_cbf_idx = {mid: i for i, mid in enumerate(movies_df['movieId'])}
            
            # Create aligned matrix matching CF shape
            aligned_content = np.zeros((item_embeddings.shape[0], raw_content_embeddings.shape[1]))
            
            # Map each CF index to its corresponding content embedding
            for cf_idx in range(item_embeddings.shape[0]):
                movie_id = mappings["idx_to_movie"].get(cf_idx)
                if movie_id and movie_id in mid_to_cbf_idx:
                    cbf_idx = mid_to_cbf_idx[movie_id]
                    if cbf_idx < raw_content_embeddings.shape[0]:
                        aligned_content[cf_idx] = raw_content_embeddings[cbf_idx]
            
            content_embeddings = aligned_content
        else:
            content_embeddings = raw_content_embeddings
        
        # Fit user embedding (CF)
        rated_indices = list(st.session_state.rated_movies.keys())
        ratings_list = list(st.session_state.rated_movies.values())
        
        user_emb, user_bias = fit_user_embedding(
            rated_indices,
            ratings_list,
            item_embeddings,
            item_biases,
            embedding_dim=EMBEDDING_DIM,
        )
        
        # Get user content profile (CBF)
        user_content_vec = get_user_content_profile(
            st.session_state.rated_movies,
            content_embeddings
        )
        
        # Generate hybrid recommendations (get more candidates for post-processing)
        candidate_recommendations = hybrid_recommend(
            user_cf_emb=user_emb,
            user_bias=user_bias,
            user_content_profile=user_content_vec,
            item_cf_embeddings=item_embeddings,
            item_biases=item_biases,
            content_embeddings=content_embeddings,
            rated_indices=set(rated_indices),
            movie_titles=mappings["movie_titles"],
            movie_genres=mappings["movie_genres"],
            alpha=HYBRID_ALPHA,
            top_k=50,  # Get more candidates for post-processing
        )
        
        # Post-process: Netflix-style weighted scoring
        # Heavily favors high-confidence matches, with diversity and novelty as secondary factors
        recommendations = post_process_recommendations(
            candidate_recommendations,
            mappings,
            top_k=10,
            trust_weight=0.7,  # High priority: relevance and confidence
            diversity_weight=0.2,  # Moderate: prevent redundancy
            novelty_weight=0.1,  # Low priority: add serendipity
        )
        
        # Save user profile
        save_user_profile(
            st.session_state.username,
            user_emb,
            user_bias,
            st.session_state.rated_movies,
            USER_DIR
        )
        
        st.session_state.recommendations = recommendations
        st.session_state.stage = "results"
        st.rerun()


def render_results_screen():
    """Render the recommendations results screen."""
    st.markdown(f"""
    <div class="app-header">
        <div class="app-title">🎉 Your Recommendations</div>
        <div class="app-subtitle">Personalized for you</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display recommendations as cards with metadata
    for i, rec in enumerate(st.session_state.recommendations, 1):
        # Handle both old format (4-tuple) and new format (5-tuple with metadata)
        if len(rec) == 5:
            idx, title, genres, score, metadata = rec
            trust = metadata.get('trust', 0)
            novelty = metadata.get('novelty', 0)
            diversity = metadata.get('diversity', 0)
            popularity = metadata.get('popularity', 0)
            avg_rating = metadata.get('avg_rating', 0)
            category = metadata.get('category', '')
        else:
            # Fallback for old format
            idx, title, genres, score = rec
            trust = novelty = diversity = 0
            popularity = 0
            avg_rating = 0
            category = ''
        
        genre_str = " | ".join(genres[:3]) if genres else "Unknown"
        
        # Create trust/diversity/novelty indicators based on category
        # Categories are determined dynamically by the Netflix-style algorithm
        trust_badge = ""
        novelty_badge = ""
        diversity_badge = ""
        
        if category == 'trust':
            trust_badge = "🛡️ High Confidence"
        elif category == 'novelty':
            novelty_badge = "✨ Hidden Gem"
        elif category == 'diversity':
            diversity_badge = "🎨 Diverse"
        # 'standard' category items don't get special badges
        
        # Build metadata string
        metadata_parts = []
        if avg_rating > 0:
            metadata_parts.append(f"⭐ {avg_rating:.1f} avg")
        if popularity > 0:
            metadata_parts.append(f"{popularity:,} ratings")
        if trust_badge:
            metadata_parts.append(trust_badge)
        if novelty_badge:
            metadata_parts.append(novelty_badge)
        if diversity_badge:
            metadata_parts.append(diversity_badge)
        
        metadata_str = " • ".join(metadata_parts) if metadata_parts else ""
        
        st.markdown(f"""
        <div class="rec-card">
            <div style="display: flex; align-items: center;">
                <span class="rec-number">{i}</span>
                <div style="flex: 1;">
                    <div class="rec-title">{title}</div>
                    <div class="rec-genre">🎭 {genre_str}</div>
                    <div style="font-size: 12px; color: #95a5a6; margin-top: 5px;">{metadata_str}</div>
                </div>
                <div class="rec-score">★ {score:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display profile info
    st.markdown(f"""
    <div style="text-align: center; color: white; margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">
        <p style="margin: 5px 0;"><strong>Profile Saved:</strong> {st.session_state.username}</p>
        <p style="margin: 5px 0;"><strong>Movies Rated:</strong> {len(st.session_state.rated_movies)}</p>
        <p style="margin: 5px 0; font-size: 12px; opacity: 0.8;">Your preferences are saved for next time!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Rate More Movies", use_container_width=True, type="primary"):
            # Keep rated movies but reset to rating stage
            st.session_state.stage = "rating"
            st.session_state.current_movie_idx = 0
            st.session_state.show_rating = False
            st.rerun()


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="CineMatch",
        page_icon="🎬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    load_custom_css()
    initialize_session_state()
    
    # Load mappings once
    if 'mappings' not in st.session_state:
        with st.spinner("Loading movie database..."):
            try:
                st.session_state.mappings = load_or_generate_mappings(DATA_DIR, MODEL_DIR)
            except Exception as e:
                st.error(f"Error loading mappings: {e}")
                st.error("Please ensure you have run `python onboarding.py` at least once to generate the mappings.")
                st.stop()
    
    mappings = st.session_state.mappings
    
    # Route to appropriate screen
    if st.session_state.stage == "welcome":
        render_welcome_screen()
    elif st.session_state.stage == "rating":
        render_rating_screen(mappings)
    elif st.session_state.stage == "generating":
        render_generating_screen()
    elif st.session_state.stage == "results":
        render_results_screen()


if __name__ == "__main__":
    main()
