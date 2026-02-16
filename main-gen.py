import pandas
import torch
import torchaudio
from torch.utils.data import DataLoader
import os
import shutil

from dataset import FMADataset
from metrics import compute_fad, compute_kld, compute_clap
from unlearning.unlearning import unl_gradient_ascent, unl_stochastic_teacher, \
    unl_one_shot_magnitude, unl_amnesiac, unl_fine_tuning
import time

from stable_audio_tools import get_pretrained_model
from config import Config
from stable_audio_tools.inference.generation import generate_diffusion_cond
import warnings

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
    output_dir = f"audio_out/tx2m/{run_id}_{Config.UNL_METHOD}/{run_id}_{Config.UNL_METHOD}_{Config.NUM_ARTISTS}artists_{stage}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Generazione {stage.upper()}: {len(test_df)} ---")

    for i, (idx, row) in enumerate(test_df.iterrows()):
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
        filename = f'sample_{i}_{genre.replace(" ", "-").replace("/","-")}_{artist.replace(" ", "-").replace("/","-")}.wav'
        filepath = os.path.join(output_dir, filename)

        torchaudio.save(filepath, audio_tensor, sample_rate)
        print(f"Salvato: {filepath}")

    return output_dir

def create_dir_real_forget(df, source_root, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    for track_id, row in df.iterrows():
        track_id_str = f"{int(track_id):06d}"
        subdir = track_id_str[:3]
        relative_path = os.path.join(subdir, f"{track_id_str}.mp3")
        source_path = os.path.join(source_root, relative_path)
        dest_path = os.path.join(target_dir, f"{track_id_str}.mp3")

        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"File {source_path} non trovato.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    start_time_total = time.time()
    #seed = numpy.random.randint(0, 2 ** 32 - 1)
    seed = 12345

    # Caricamento Dati e modello
    tracks = pandas.read_csv(Config.CSV_FILE, index_col=0, header=[0, 1])
    track_infos = tracks[[('track', 'genre_top'), ('artist', 'name')]].dropna()

    tracks_to_remove = [
        1486, 2624, 3284, 5574, 8669, 10116, 11583, 12838, 13529, 14116, 14180, 20814, 22554, 23429, 23430,
        23431, 25173, 25174, 25175, 25176, 25180, 29345, 29346, 29352, 29356, 33411, 33413, 33414, 33417,
        33418, 33419, 33425, 35725, 39363, 41745, 42986, 43753, 50594, 50782, 53668, 54569, 54582, 61480,
        61822, 63422, 63997, 65753, 72656, 72980, 73510, 80391, 80553, 82699, 84503, 84504, 84522, 84524,
        86656, 86659, 86661, 86664, 87057, 90244, 90245, 90247, 90248, 90250, 90252, 90253, 90442, 90445,
        91206, 92479, 94052, 94234, 95253, 96203, 96207, 96210, 98105, 98558, 98559, 98560, 98562, 98571,
        99134, 101265, 101272, 101275, 102241, 102243, 102247, 102249, 102289, 105247, 106409, 106412,
        106415, 106628, 108920, 108925, 109266, 110236, 115610, 117441, 126981, 127336, 127928, 129207,
        129800, 130328, 130748, 130751, 131545, 133297, 133641, 133647, 134887, 140449, 140450, 140451,
        140452, 140453, 140454, 140455, 140456, 140457, 140458, 140459, 140460, 140461, 140462, 140463,
        140464, 140465, 140466, 140467, 140468, 140469, 140470, 140471, 140472, 142614, 143992, 144518,
        144619, 145056, 146056, 147419, 147424, 148786, 148787, 148788, 148789, 148790, 148791, 148792,
        148793, 148794, 148795, 151920, 155051, 134956
    ]

    # Rimuovi i track problematici dall'indice
    track_infos = track_infos.drop(tracks_to_remove, errors='ignore')

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model.to(Config.DEVICE)

    forget_df, retain_df, chosen_artists = create_forget_set_by_artist(track_infos, n_artists=Config.NUM_ARTISTS)
    print(f"Artisti da dimenticare: {chosen_artists}")

    real_dir = "../data/forget_set"
    create_dir_real_forget(forget_df, "../data/fma_large", real_dir)

    pre_dir = "audio_out/tx2m/50artists_generation_pre"
    #pre_dir = generate_samples_from_metadata(model, model_config, forget_df, stage="pre", run_id=Config.TIMESTAMP)

    # PREPARAZIONE DATALOADERS PER UNLEARNING
    forget_dataset = FMADataset(forget_df.index, metadata_df=tracks)
    retain_dataset = FMADataset(retain_df.index, metadata_df=tracks)

    forget_loader = DataLoader(forget_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # UNLEARNING METHODS
    print(f"Inizio Unlearning con metodo {Config.UNL_METHOD}...")

    if Config.UNL_METHOD == "FT":
        unl_model = unl_fine_tuning(model, forget_loader, retain_loader, epochs=Config.EPOCHS, lr=Config.LR,
                                    lambda_unlearn=0.5)
    elif Config.UNL_METHOD == "GA":
        unl_model = unl_gradient_ascent(model, forget_loader, retain_loader, epochs=Config.EPOCHS, lr=Config.LR,
                                        alpha=1, beta=1)
    elif Config.UNL_METHOD == "ST":
        unl_model = unl_stochastic_teacher(model, forget_loader, retain_loader, epochs=Config.EPOCHS, lr=Config.LR,
                                           alpha=0.01, beta=0.01)
    elif Config.UNL_METHOD == "OSM":
        unl_model = unl_one_shot_magnitude(model, threshold=0.1)
    elif Config.UNL_METHOD == "A":
        unl_model = unl_amnesiac(model, forget_loader, lr=Config.LR)
    else:
        print("unknown method")

    # GENERAZIONE POST-UNLEARNING
    post_dir = generate_samples_from_metadata(unl_model.model, model_config, forget_df, stage="post",
                                              run_id=Config.TIMESTAMP)
    print("PRE DIR:")
    print(pre_dir)
    print("POST DIR:")
    print(post_dir)

    # METRICS
    fad_pre = compute_fad(real_dir, pre_dir)
    fad_post = compute_fad(real_dir, post_dir)

    kld_pre = compute_kld(real_dir, pre_dir)
    kld_post = compute_kld(real_dir, post_dir)

    clap_pre = compute_clap(pre_dir)
    clap_post = compute_clap(post_dir)

    # PRINT RESULTS
    print("\nCONFIGS:")
    print(f"Artists to forget: {Config.NUM_ARTISTS}")
    print(f"Songs to forget: {len(forget_df)}")
    print(f"Unlearn Method: {Config.UNL_METHOD}")
    print(f"Learning Rate: {Config.LR}")
    print(f"Epochs: {Config.EPOCHS}")

    duration = time.time() - start_time_total
    print(f"\nRUNNING TIME: {duration / 60:.2f} minutes")

    print("\n\t\tPRE\t\tPOST\t\tDIFF")
    print(f"FAD\t\t{fad_pre:.3f}\t\t\t{fad_post:.3f}\t\t\t{fad_post - fad_pre:.3f}")
    print(f"KLD\t\t{kld_pre:.3f}\t\t\t{kld_post:.3f}\t\t\t{kld_post - kld_pre:.3f}")
    print(f"CLAP\t{clap_pre:.3f}\t\t\t{clap_post:.3f}\t\t\t{clap_post - clap_pre:.3f}\n")
