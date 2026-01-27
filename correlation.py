import os
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# =========================
# CONFIG
# =========================
FMA_AUDIO_DIR = "../data/fma_large"
FMA_METADATA = "../data/tracks.csv"
GENERATED_TRACK = "audio_out/pretrain_model/260111_150609_rock_music.wav"

N_MFCC = 20
SR = 22050

ID_CSV = "stable_audio_ids.csv"

df_ids = pd.read_csv(ID_CSV)

# insieme per controllo veloce
valid_ids = set(df_ids["track_id"].astype(int))


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)
        y = y[:SR * 30]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        return np.mean(mfcc, axis=1)
    except:
        return None


# =========================
# LOAD METADATA
# =========================
tracks = pd.read_csv(FMA_METADATA, index_col=0, header=[0, 1])

def get_metadata(track_id):
    try:
        artist = tracks.loc[track_id, ('artist', 'name')]
        title = tracks.loc[track_id, ('track', 'title')]
        genre = tracks.loc[track_id, ('track', 'genre_top')]
        return artist, title, genre
    except:
        return "Unknown", "Unknown", "Unknown"

# =========================
# FEATURE OF GENERATED TRACK
# =========================
generated_feat = extract_features(GENERATED_TRACK)
generated_feat = generated_feat.reshape(1, -1)

# =========================
# LOOP OVER FMA
# =========================
results = []
i = 0
for root, _, files in os.walk(FMA_AUDIO_DIR):
    for file in files:

        try:
            track_id = int(os.path.splitext(file)[0])
        except ValueError:
            continue

        if track_id not in valid_ids:
            continue
        i = i + 1
        print(f"{i} / 13874")
        if not file.endswith((".mp3", ".wav")):
            continue

        path = os.path.join(root, file)

        feat = extract_features(path)
        if feat is None:
            continue

        similarity = cosine_similarity(
            generated_feat,
            feat.reshape(1, -1)
        )[0][0]

        artist, title, genre = get_metadata(track_id)

        results.append({
            "track_id": track_id,
            "similarity": similarity,
            "artist": artist,
            "title": title,
            "genre": genre
        })

# =========================
# SORT & PRINT
# =========================

results = sorted(results, key=lambda x: x["similarity"], reverse=True)

print("\nBrani più correlati:\n")
for r in results[:20]:
    print(
        f"{r['similarity']:.3f} | {r['artist']} - {r['title']} ({r['genre']})"
    )
