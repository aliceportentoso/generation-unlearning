import os
import pandas as pd

# PATH HARD-CODED (semplice, funziona)
CSV_PATH = "../data/tracks.csv"

# caricato una sola volta per worker
df = pd.read_csv(
    CSV_PATH,
    index_col=0,
    header=[0, 1]
)

def get_custom_metadata(info, audio):
    # 1. estrai track_id dal nome file
    filename = os.path.basename(info["path"])
    track_id = int(os.path.splitext(filename)[0])

    # 2. fallback se manca
    if track_id not in df.index:
        return {"__reject__": True}

    row = df.loc[track_id]

    # 3. ritorna SOLO quello che serve al conditioning
    return {"prompt": f"Artist: {str(row[("artist", "name")])}, Genre: {str(row[("track", "genres")])}"}