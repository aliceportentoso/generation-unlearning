import numpy
import pandas
import torch

from stable_audio_tools import get_pretrained_model
from config import Config
import joblib

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.inference.inference import save_audio

balance = False

def create_forget_set(df, n_samples=500):
    # 1. Estrazione casuale del forget set
    # .sample() è più sicuro di np.random.choice per i DataFrame
    forget_set = df.sample(n=n_samples, random_state=42)

    # 2. Creazione del retain set escludendo gli indici selezionati
    retain_set = df.drop(forget_set.index)

    # Restituiamo i DataFrame completi, così hai accesso a tutti i campi
    return forget_set, retain_set

def generate_samples_from_metadata(model, model_config, forget_artists, stage="pre"):
    """
    forget_artists: lista di stringhe (prompt) associate ai 500 brani
    """
    model.eval()
    device = next(model.parameters()).device
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    print(f"Generazione in corso: {stage} unlearning...")

    for i, artist in enumerate(forget_artists):
        # 1. Creiamo la lista di condizionamento (obbligatoria)
        # Stable Audio Open si aspetta spesso prompt e parametri come secondi di audio
        prompt = f"generate a song in the style of {artist}"
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": 30  # o la durata prevista nel tuo config
        }]

        # 2. Chiamata corretta
        # Passiamo 'conditioning' esplicitamente come previsto dalla libreria
        audio = generate_diffusion_cond(
            model,
            steps=50,
            cfg_scale=7.0,
            conditioning=conditioning,  # <--- CORREZIONE: Usa conditioning invece di conditioning_config
            sample_size=sample_size,
            device=device
        )
        audio = audio.detach().cpu().squeeze(0)

        # Salva o accumula l'audio
        save_audio(f"audio_out/prompt/{stage}/sample_{i}_{artist}.wav", audio, sample_rate)
        print(f"audio file saved in audio_out/prompt/{stage}/sample_{i}_{artist}.wav")

tracks = pandas.read_csv(Config.CSV_FILE,  index_col=0, header=[0,1])

sub_tracks = tracks[tracks[('set', 'subset')].isin(["small", "medium"])] #106 mila
track_infos = tracks[[('track', 'genre_top'),('artist','name'),('album','information')]].dropna() #49 mila

# --- Esempio di utilizzo ---
forget_df, retain_df = create_forget_set(track_infos)

# Se ti servono solo gli ID (gli indici):
forget_ids = forget_df.index

# Carica il Modello Stable Audio Open
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")

# GENERAZIONE PRE UNLEARNING
# Recupera i prompt testuali per i f_ids (necessita di accesso ai metadati FMA)
forget_artists = track_infos.loc[forget_ids, ('artist', 'name')].tolist()

generate_samples_from_metadata(model, model_config, forget_artists, stage="pre")

# UNLEARNING

# GENERAZIONE POST-UNLEARNING
generate_samples_from_metadata(model, model_config, forget_artists, stage="post")
