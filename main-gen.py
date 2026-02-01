import os
import numpy
import pandas
import torch
import torchaudio
from torch.utils.data import DataLoader

from dataset import FMADataset
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper
from unlearning.unlearning import unl_fine_tuning, unl_fine_tuning_gradient_ascent, unl_stochastic_teacher, \
    unl_one_shot_magnitude, unl_amnesiac
import time

from stable_audio_tools import get_pretrained_model
from config import Config
from stable_audio_tools.inference.generation import generate_diffusion_cond

def create_forget_set(df, n_samples):
    # 1. Estrazione casuale del forget set
    forget_set = df.sample(n=n_samples, random_state=seed)

    # 2. Creazione del retain set escludendo gli indici selezionati
    retain_set = df.drop(forget_set.index)

    # Restituiamo i DataFrame completi, così hai accesso a tutti i campi
    return forget_set, retain_set

def create_forget_set_by_artist(df, n_artists):
    unique_artists = df[('artist', 'name')].unique()
    artists_to_forget = pandas.Series(unique_artists).sample(n=n_artists, random_state=seed).values

    forget_mask = df[('artist', 'name')].isin(artists_to_forget)
    forget_set = df[forget_mask]
    print(f"Dim forget: {len(forget_set)}")
    retain_set = df[~forget_mask]

    return forget_set, retain_set, artists_to_forget


def generate_samples_from_metadata(model, model_config, forget_df, stage="pre", run_id=""):
    model.eval()
    device = Config.DEVICE
    sample_rate = model_config["sample_rate"]

    test_df = forget_df.drop_duplicates(subset=[('artist', 'name')])

    # Cartella di output pulita
    output_dir = f"audio_out/tx2m/{run_id}_{Config.UNL_METHOD}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Generazione {stage.upper()}: {len(test_df)} ---")

    # Cicliamo sulle righe del dataframe per avere prompt specifici
    for i, (idx, row) in enumerate(test_df.iterrows()):
        artist = row[('artist', 'name')]
        genre = row[('track', 'genre_top')]

        # Prompt specifico per testare l'unlearning
        prompt = f"{genre} song in the style of {artist}"

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": 30
        }]

        model.to(device)
        if "PeftModel" in str(type(model)):
            mod = model.model
        elif not isinstance(model, ConditionedDiffusionModelWrapper):
            mod = model.model
        else:
            mod = model

        with torch.no_grad():
            audio = generate_diffusion_cond(
                model = mod,
                steps=50,
                cfg_scale=7.0,
                conditioning=conditioning,
                sample_size=model_config["sample_size"],
                device=device,
                seed=seed
            )

        # Trasformazione in formato salvabile
        audio_tensor = audio.detach().cpu().squeeze(0)  # [canali, campioni]

        filename = f'sample_{i}_{artist.replace(" ", "_").replace("/","_")}_{stage}.wav'
        filepath = os.path.join(output_dir, filename)

        torchaudio.save(filepath, audio_tensor, sample_rate)
        print(f"Salvato: {filepath}")

if __name__ == "__main__":
    start_time_total = time.time()
    run_timestamp = time.strftime("%Y%m%d-%H%M")
    seed = numpy.random.randint(0, 2 ** 32 - 1)

    # 1. Caricamento Dati
    tracks = pandas.read_csv(Config.CSV_FILE, index_col=0, header=[0, 1])
    track_infos = tracks[[('track', 'genre_top'), ('artist', 'name')]].dropna()

    # 2. Split Forget/Retain
    forget_df, retain_df, chosen_artists = create_forget_set_by_artist(track_infos, n_artists=Config.NUM_ARTISTS)
    print(f"Artisti da dimenticare: {chosen_artists}")

    # 3. Caricamento Modello
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model.to(Config.DEVICE)

    ####torch.cuda.empty_cache()
    ####gc.collect()

    # 4. GENERAZIONE PRE-UNLEARNING
    # Usiamo il dataframe per generare basandoci sui metadati reali
    generate_samples_from_metadata(model, model_config, forget_df, stage="pre", run_id=run_timestamp)

    # 5. PREPARAZIONE DATALOADERS PER UNLEARNING
    # Usiamo la classe FMADataset definita precedentemente che carica l'audio reale
    forget_dataset = FMADataset(forget_df.index, [0] * len(forget_df), metadata_df=tracks)
    retain_dataset = FMADataset(retain_df.index.to_series().sample(n=1000), [0] * 1000,
                                metadata_df=tracks)  # Esempio: 1000 brani di retain

    forget_loader = DataLoader(forget_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 6. UNLEARNING
    # Ora passiamo i loader, non i dataframe
    print(f"Inizio Unlearning con metodo {Config.UNL_METHOD}...")

    if Config.UNL_METHOD == "GA":
        unl_model = unl_fine_tuning_gradient_ascent(model, forget_loader, retain_loader, model_config, epochs=Config.EPOCHS, lr=Config.LR)
    elif Config.UNL_METHOD == "ST":
        unl_model = unl_stochastic_teacher(model, forget_loader, epochs=Config.EPOCHS, lr=Config.LR/2)
    elif Config.UNL_METHOD == "OSM":
        unl_model = unl_one_shot_magnitude(model, threshold=0.1)
    elif Config.UNL_METHOD == "A":
        unl_model = unl_amnesiac(model, forget_loader, lr=Config.LR)
    else:
        print("unknown method")


    # 7. GENERAZIONE POST-UNLEARNING
    generate_samples_from_metadata(unl_model.base_model, model_config, forget_df, stage="post", run_id=run_timestamp)

    duration = time.time() - start_time_total
    print(f"Esecuzione completata in: {duration / 60:.2f} minuti")