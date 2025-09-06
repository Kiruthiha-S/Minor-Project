import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = "kiruthiha.csv"
df = pd.read_csv(CSV_PATH)

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

track_col = pick_col(df, ["track_name", "name", "track"])
artist_col = pick_col(df, ["artist", "artists", "artist_name", "track_artist"])
genre_col = pick_col(df, ["playlist_genre", "genre"])
plist_col = pick_col(df, ["playlist_name", "playlist", "playlist_title"])
numeric_candidates = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]
numeric_cols = [c for c in numeric_candidates if c in df.columns]
df = df.copy()
df = df.dropna(subset=numeric_cols).reset_index(drop=True)
to_encode = [c for c in [genre_col, plist_col] if c is not None]
df_encoded = pd.get_dummies(df, columns=to_encode, drop_first=False)
scaler = StandardScaler()
df_encoded.loc[:, numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
feature_cols = [c for c in df_encoded.columns if df_encoded[c].dtype != "object"]
X = df_encoded[feature_cols].values
    
print("\n--- 1. Data Pre-processing ---")
try:
    df = pd.read_csv("kiruthiha.csv")
    print("Dataset loaded successfully.\n")
    print("First 5 rows of data:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'kiruthiha.csv' not found. ")
    exit()
df.columns = df.columns.str.strip().str.lower()
if "duration_ms" in df.columns:
    df["duration_s"] = df["duration_ms"] / 1000
    df.drop("duration_ms", axis=1, inplace=True)
elif "duration" in df.columns:
    df["duration_s"] = df["duration"] / 1000
    df.drop("duration", axis=1, inplace=True)
else:
    print("skip duration processing.")

print("\n--- Cleaned Columns ---")
print(df.columns)

print("\n--- 2. Exploratory Data Analysis ---")
if "artists" in df.columns:
    top_artists = df["artists"].value_counts().head(10)
    top_artists.plot(kind="bar", figsize=(8, 5), color="skyblue")
    plt.title("Top 10 Artists by Song Count")
    plt.xlabel("Artist")
    plt.ylabel("Number of Songs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
if "popularity" in df.columns:
    plt.hist(df["popularity"], bins=20, color="lightgreen", edgecolor="black")
    plt.title("Popularity Distribution")
    plt.xlabel("Popularity Score")
    plt.ylabel("Number of Songs")
    plt.show()

print("\nAnalysis successfully!")
corr_matrix = df[numeric_cols].corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.title("PCA Clusters")
plt.tight_layout()
plt.show()
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df["cluster"] = clusters
if {"energy", "danceability"}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="energy", y="danceability", hue="cluster", palette="viridis")
    plt.title("Clusters based on Energy & Danceability")
    plt.tight_layout()
    plt.show()
def recommend_songs(song_name, num_recommendations=5):
    if track_col is None:
        return "Track name column not found in dataset."

    match_idx = df.index[df[track_col].astype(str).str.lower() == str(song_name).lower()].tolist()
    if not match_idx:
        return "Song not found!"
    i = match_idx[0]
    sims = cosine_similarity(X[i].reshape(1, -1), X)[0]
    sims[i] = -np.inf
    top_idx = np.argsort(sims)[-num_recommendations:][::-1]
    cols_to_show = [c for c in [track_col, artist_col, genre_col, "cluster"] if c is not None and c in df.columns]
    result = df.loc[top_idx, cols_to_show].reset_index(drop=True)
    return result
