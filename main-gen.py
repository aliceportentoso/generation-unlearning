import numpy
import pandas
import torch
import torchaudio
from torch.utils.data import DataLoader

from dataset import FMADataset
from metrics import compute_fad, compute_kld, compute_clap
from unlearning.unlearning import unl_gradient_ascent, unl_stochastic_teacher, \
    unl_one_shot_magnitude, unl_amnesiac, unl_fine_tuning
import time

from stable_audio_tools import get_pretrained_model
from config import Config
from stable_audio_tools.inference.generation import generate_diffusion_cond

def create_forget_set(df, n_samples):
    forget_set = df.sample(n=n_samples, random_state=seed)
    retain_set = df.drop(forget_set.index)
    return forget_set, retain_set

def create_forget_set_by_artist(df, n_artists):
    unique_artists = df[('artist', 'name')].unique()
    artists_to_forget = pandas.Series(unique_artists).sample(n=n_artists, random_state=seed).values

    forget_mask = df[('artist', 'name')].isin(artists_to_forget)
    forget_set = df[forget_mask]
    print(f"Dim forget: {len(forget_set)}")
    retain_set = df[~forget_mask]

    return forget_set, retain_set, artists_to_forget

def generate_samples_from_metadata(model, model_config, forget_df, stage, run_id=""):
    model.eval()
    device = Config.DEVICE
    sample_rate = model_config["sample_rate"]

    test_df = forget_df.drop_duplicates(subset=[('artist', 'name')])

    output_dir = f"audio_out/tx2m/{run_id}_{Config.UNL_METHOD}_{stage}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Generazione {stage.upper()}: {len(test_df)} ---")

    for i, (idx, row) in enumerate(test_df.head(1).iterrows()):
        artist = row[('artist', 'name')]
        genre = row[('track', 'genre_top')]
        prompt = f"{genre} song in the style of {artist}"

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": 30
        }]

        model.to(device)

        with torch.no_grad():
            audio = generate_diffusion_cond(
                model = model,
                steps=50,
                cfg_scale=7.0,
                conditioning=conditioning,
                sample_size=model_config["sample_size"],
                device=device,
                seed=seed
            )

        audio_tensor = audio.detach().cpu().squeeze(0)  # [canali, campioni]

        filename = f'sample_{i}_{artist.replace(" ", "_").replace("/","_")}.wav'
        filepath = os.path.join(output_dir, filename)

        torchaudio.save(filepath, audio_tensor, sample_rate)
        print(f"Salvato: {filepath}")

    return output_dir


import os
import shutil

def create_dir_real_forget(df, source_root, target_dir):

    # 1. Crea la directory di destinazione se non esiste
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)

    count = 0
    errors = 0

    print("Copia dei brani in corso...")
    for track_id, row in df.iterrows():
        track_id_str = f"{int(track_id):06d}"

        # 3. Costruisci il path relativo: le prime 3 cifre sono la cartella
        # Esempio: 114577 -> cartella '114', file '114577.mp3'
        subdir = track_id_str[:3]
        relative_path = os.path.join(subdir, f"{track_id_str}.mp3")

        # 4. Path sorgente completo
        source_path = os.path.join(source_root, relative_path)

        # 5. Path destinazione (copiamo tutto "piatto" nella nuova cartella)
        dest_path = os.path.join(target_dir, f"{track_id_str}.mp3")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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

    real_dir = "../data/forget_set"
    create_dir_real_forget(forget_df, "../data/fma_large", real_dir)

    # 4. GENERAZIONE PRE-UNLEARNING
    # Usiamo il dataframe per generare basandoci sui metadati reali
    pre_dir = generate_samples_from_metadata(model, model_config, forget_df, stage="pre", run_id=run_timestamp)

    # 5. PREPARAZIONE DATALOADERS PER UNLEARNING
    forget_dataset = FMADataset(forget_df.index, metadata_df=tracks)
    retain_dataset = FMADataset(retain_df.index, metadata_df=tracks)

    forget_loader = DataLoader(forget_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 6. UNLEARNING
    print(f"Inizio Unlearning con metodo {Config.UNL_METHOD}...")

    if Config.UNL_METHOD == "FT":
        unl_model = unl_fine_tuning(model, forget_loader, retain_loader, epochs=Config.EPOCHS, lr=Config.LR,
                                    lambda_unlearn=1.5)
    elif Config.UNL_METHOD == "GA":
        unl_model = unl_gradient_ascent(model, forget_loader, retain_loader, epochs=Config.EPOCHS, lr=Config.LR,
                                        alpha=0.01)
    elif Config.UNL_METHOD == "ST":
        unl_model = unl_stochastic_teacher(model, forget_loader, retain_loader, epochs=Config.EPOCHS, lr=Config.LR,
                                           alpha=0.01, beta=0.01)
    elif Config.UNL_METHOD == "OSM":
        unl_model = unl_one_shot_magnitude(model, threshold=0.1)
    elif Config.UNL_METHOD == "A":
        unl_model = unl_amnesiac(model, forget_loader, lr=Config.LR)
    else:
        print("unknown method")

    # 7. GENERAZIONE POST-UNLEARNING
    post_dir = generate_samples_from_metadata(unl_model.model, model_config, forget_df, stage="post",
                                              run_id=run_timestamp)

    fad_pre = compute_fad(real_dir, pre_dir)
    print(f"FAD pre = {fad_pre}")
    fad_post = compute_fad(real_dir, post_dir)
    print(f"FAD post = {fad_post}")

    kld_pre = compute_kld(real_dir, pre_dir)
    print(f"KLD pre = {kld_pre}")
    kld_post = compute_kld(real_dir, post_dir)
    print(f"KLD post = {kld_post}")

    clap_pre = compute_clap(pre_dir, forget_df)
    print(f"CLAP pre = {clap_pre}")
    clap_post = compute_clap(post_dir, forget_df)
    print(f"CLAP post = {clap_post}")

    print(f"DIFFERENZA DI FAD: {fad_post - fad_pre}")
    print(f"DIFFERENZA DI KLD: {kld_post - kld_pre}")
    print(f"DIFFERENZA DI CLAP: {clap_post - clap_pre}")

    duration = time.time() - start_time_total
    print(f"Esecuzione completata in: {duration / 60:.2f} minuti")