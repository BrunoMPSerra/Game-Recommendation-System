import streamlit as st
import pickle
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from youtubesearchpython import VideosSearch
from googleapiclient.discovery import build

# Load saved model, embeddings, and dataset
with open("saved_model.pkl", "rb") as f:
    model_data = pickle.load(f)

bert_model = model_data["bert_model"]  # Load BERT model
game_embeddings = model_data["game_embeddings"]  # Load embeddings
kmeans = model_data["kmeans"]  # Load K-Means model
df_games = model_data["df_games"]  # Load dataset

# Configure Gemini AI
genai.configure(api_key="INPUT YOUR API KEY HERE")


# ------------------------------- FUNCTIONS -------------------------------

def interpret_user_input(user_input):
    """ Get recommended game titles from Gemini AI """
    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = f"""
    Based on the following user request:
    "{user_input}"
    
    Provide a list of **30** game titles that best match the request.
    Only return the game titles, one per line, in the following format:
    
    - Game Title 1
    - Game Title 2
    - Game Title 3
    ...
    - Game Title 30

    Do NOT provide descriptions, explanations, or additional text.
    """

    response = model.generate_content(prompt)
    return response.text.strip()


def extract_game_titles(gemini_response):
    """ Extract only game titles from Gemini AI response """
    titles = re.findall(r"- (.+)", gemini_response)
    return titles


def filter_games_in_dataset(recommended_titles, df_games):
    """ Find recommended games that exist in our dataset """
    recommended_titles_lower = [title.lower() for title in recommended_titles]
    valid_games = df_games[df_games["title_lower"].isin(recommended_titles_lower)][["title", "summary"]]
    return valid_games


def rank_games_by_similarity(user_query, filtered_games, game_embeddings, bert_model, df_games):
    """ Rank games using BERT similarity """
    if filtered_games.empty:
        return None

    query_embedding = bert_model.encode([user_query], convert_to_numpy=True)

    filtered_indices = df_games[df_games["title"].isin(filtered_games["title"])].index.tolist()

    if max(filtered_indices) >= len(game_embeddings):
        return None

    filtered_embeddings = game_embeddings[filtered_indices]
    similarities = cosine_similarity(query_embedding, filtered_embeddings)

    filtered_games = filtered_games.copy()
    filtered_games["similarity"] = similarities[0]

    ranked_games = filtered_games.sort_values("similarity", ascending=False).drop_duplicates(subset="title")

    return ranked_games[["title", "summary", "similarity"]]

YOUTUBE_API_KEY = "INPUT YOUR API KEY HERE"

def get_youtube_trailer(game_title):
    """Fetch the first YouTube trailer using YouTube Data API."""
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    
    request = youtube.search().list(
        q=f"{game_title} official game trailer",
        part="snippet",
        type="video",
        maxResults=1
    )
    
    response = request.execute()
    
    if "items" in response and len(response["items"]) > 0:
        video_id = response["items"][0]["id"]["videoId"]
        return f"https://www.youtube.com/watch?v={video_id}"
    
    return None


# ------------------------------- STREAMLIT UI -------------------------------

# Page Layout
st.set_page_config(page_title="Game Recommendation System", page_icon="🎮", layout="wide")

# Title and Description
st.markdown("<h1 style='text-align: center;'>🎮 AI-Powered Game Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Enter a game description, game title, or genre to get recommendations.</p>", unsafe_allow_html=True)

# Input box for user query
user_query = st.text_input("🎯 **Describe the type of game you are looking for:**", placeholder="E.g., Open-world RPG like Skyrim")

# Button
if st.button("🔍 Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        structured_query = interpret_user_input(user_query)
        recommended_titles = extract_game_titles(structured_query)
        recommended_titles = [re.sub(r"^\d+\.\s*", "", title).strip() for title in recommended_titles]

        if not recommended_titles:
            st.error("⚠️ No recommendations received from Gemini.")
            st.stop()

        # Filter dataset for valid games (background operation)
        filtered_games = filter_games_in_dataset(recommended_titles, df_games)

        if filtered_games.empty:
            st.error("⚠️ No games from Gemini's recommendations exist in the dataset.")
            st.stop()

        # Rank games based on similarity
        ranked_games = rank_games_by_similarity(user_query, filtered_games, game_embeddings, bert_model, df_games)

        # Display the final recommendations
        if ranked_games is not None and not ranked_games.empty:
            st.subheader("🕹️ **Top Game Recommendations Based on Your Input:**")

            # Display recommendations using expanders
            for index, row in ranked_games.iterrows():
                with st.expander(f"🎮 {row['title']}"):
                    st.write(f"**Summary:** {row['summary']}")

                    # Fetch and display YouTube trailer
                    trailer_url = get_youtube_trailer(row['title'])
                    if trailer_url:
                        st.video(trailer_url)
                    else:
                        st.write("🎥 No trailer found.")
        else:
            st.error("❌ No valid game recommendations found.")


