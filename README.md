🕹️ AI-Powered Game Recommendation System

An advanced game recommendation system powered by AI, Machine Learning, and NLP to suggest games based on user input.


📌 Overview

This project is an AI-driven Game Recommendation System that allows users to describe the type of game they want to play, and it intelligently suggests relevant games based on their preferences.
The system leverages Google Gemini AI for generating game recommendations, BERT embeddings + K-Means clustering for understanding game descriptions, and cosine similarity for ranking recommendations.
Additionally, it features a Streamlit web application where users can interact with the system and even watch game trailers from YouTube.


🎯 Features
✅ AI-Powered Recommendations - Uses Google Gemini AI to generate relevant game suggestions.
✅ Machine Learning-Based Ranking - BERT embeddings + cosine similarity provide intelligent recommendations.
✅ Streamlit UI - A user-friendly web application for game searches and recommendations.
✅ Game Description Matching - Uses BERT embeddings to understand and compare game descriptions.
✅ YouTube Trailer Integration - Automatically fetches trailers for the recommended games.
✅ Pre-Trained Model for Faster Execution - Uses pickle to load the trained model instantly.


📂 Project Structure

📦 AI-Powered Game Recommendation System
│-- 📜 README.md                <- Project documentation (this file)
│-- 📜 app2.0.py                <- Streamlit web application
│-- 📜 Game Recommendation System.ipynb <- Jupyter Notebook (model training & experiments)
│-- 📜 cleaned_games.csv         <- Cleaned dataset after preprocessing
│-- 📜 games.csv                 <- Raw dataset
│-- 📜 saved_model.pkl           <- Trained BERT + K-Means model


📊 Dataset Information

The dataset used in this project consists of video game metadata such as:
Title (Game name)
Summary (Game description)
Genres (Action, RPG, Adventure, etc.)
Release Year
Platform (PC, PlayStation, Xbox, etc.)
Other metadata (Developer, Publisher, etc.)

Data Preprocessing Steps:
Cleaned missing values (Dropped empty summaries/titles)
Lowercased text (Standardization)
Extracted year from release date
Combined multiple text fields for better NLP analysis
Stored cleaned dataset as cleaned_games.csv


🧠 Machine Learning Techniques Used

This project implements various AI and ML techniques to provide accurate recommendations:
1️⃣ Natural Language Processing (NLP)
Text Embeddings using BERT: Converts game descriptions into numerical vectors using sentence-transformers.
2️⃣ Clustering for Efficient Recommendations
K-Means Clustering: Groups similar games into clusters for faster and more relevant filtering.
Why K-Means?: Helps narrow down recommendations based on game similarities.
3️⃣ AI-Powered Suggestions
Google Gemini AI: Converts user input into structured queries and generates relevant game titles.
4️⃣ Game Ranking System
Cosine Similarity: Compares the user’s request with game descriptions and ranks recommendations.
Pre-trained BERT embeddings allow for contextual understanding of descriptions.
5️⃣ YouTube API for Game Trailers
YouTube Data API v3 fetches official game trailers based on game titles.


🖥️ Web Application (Streamlit)

The front-end of the recommendation system is built using Streamlit, providing:
User Input Box - Where users enter game descriptions (e.g., "I want an open-world RPG like Skyrim").
AI-Generated Recommendations - Lists relevant game titles based on user input.
Game Summaries - Brief descriptions of the recommended games.
YouTube Trailers - Embeds trailers for recommended games.

How to Run the App Locally
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app2.0.py
Open the localhost URL in your browser to use the app.


📈 Key Findings

BERT embeddings significantly improve game recommendations compared to traditional text similarity approaches.
K-Means clustering speeds up recommendation retrieval while maintaining accuracy.
Google Gemini AI generates diverse and relevant game suggestions, even for niche user queries.
Cosine similarity effectively ranks recommendations, ensuring the most relevant games are shown first.
YouTube trailer integration enhances user engagement, making recommendations more interactive.


🚀 Future Improvements

Advanced Filtering: Allow users to filter by genre, platform, multiplayer/single-player, etc.
Faster Recommendations: Replace K-Means with FAISS (Facebook AI Similarity Search) for real-time retrieval.
UI Enhancements: Improve design with images, ratings, and additional metadata.
Deployment: Host the Streamlit app on Streamlit Cloud, Hugging Face Spaces, or Google Cloud.
Deep Learning Model: Explore fine-tuned transformers to improve recommendation quality.


👨‍💻 Technologies Used

Python 
Streamlit (Web UI)
Google Gemini AI API (Natural Language Processing)
BERT (sentence-transformers) (Text embeddings)
K-Means Clustering (sklearn) (Game similarity grouping)
Cosine Similarity (sklearn) (Ranking system)
YouTube Data API v3 (Game trailer fetching)
Pandas & NumPy (Data processing)


🎮 Final Thoughts
This AI-powered game recommendation system combines cutting-edge NLP, ML, and AI techniques to provide accurate, context-aware suggestions for gamers. The integration of Google Gemini AI, BERT embeddings, and YouTube trailers creates a unique and engaging experience for users looking for new games.

Next Steps: Deploy the app and make it available for public use! 


Made by Bruno Serra

For questions or contributions, feel free to reach out! 🎮🚀