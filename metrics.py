import torch.nn.functional as F
from hear21passt.base import get_basic_model
from scipy.linalg import sqrtm
import torchaudio
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from CLAPScore_for_LASS_main.models.clap_encoder import CLAP_Encoder
from config import Config

import os
import torch
import numpy as np
from frechet_audio_distance import FrechetAudioDistance
from scipy.stats import entropy

def compute_kld(real_path, gen_path, n_bins=50, eps=1e-10):
    model = get_basic_model(mode="logits").to(Config.DEVICE)
    model.eval()

    p_dist = get_dataset_distribution(real_path, model)
    q_dist = get_dataset_distribution(gen_path, model)

    # Evita log(0)
    P = np.clip(p_dist, eps, 1.0)
    Q = np.clip(q_dist, eps, 1.0)

    # Normalizza
    P /= np.sum(P)
    Q /= np.sum(Q)

    # KL divergence
    kld = entropy(P, Q)
    return kld


def compute_fad(real_path, gen_path):
    fad = FrechetAudioDistance(model_name="vggish", sample_rate=16000)

    # --- Calcolo FAD ---
    score = fad.score(real_path, gen_path)

    return score


def compute_clap(folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_checkpoint = "music_speech_audioset_epoch_15_esc_89.98.pt"
    model = CLAP_Encoder(device=device, pretrained_path=pretrained_checkpoint).eval()

    scores_dict = {}
    scores_list = []

    with torch.no_grad():
        for filename in tqdm(os.listdir(folder_path)):
            if not filename.endswith(".wav"):
                continue

            audio_path = os.path.join(folder_path, filename)
            prompt_text = filename.replace(".wav", "").replace("_", " ").replace("-", " ")

            if filename.endswith(".wav"):
                name_parts = filename.replace(".wav", "").replace("-", " ").split("_")
                genre = "".join(name_parts[2])
                artist = " ".join(name_parts[3:])
                print("\ngenre, artist = ")
                print(genre, artist)
                prompt = f"{genre} song in the style of {artist}"
            # --- Carica audio ---
            waveform, sr = torchaudio.load(audio_path)  # [channels, samples]

            # Se stereo, media sui canali
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Sposta su device
            waveform = waveform.to(device)

            # --- DEBUG ---
            print("DEBUG: Prompt text:", prompt_text)
            print("DEBUG: Waveform original shape:", waveform.shape)
            print("DEBUG: Waveform numel:", waveform.numel())

            # --- Prepara embedding audio ---
            if waveform.numel() == 0:
                print(f"WARNING: Audio file {filename} is empty, skipping...")
                continue

            # Rimuove solo l’asse dei canali se c’è, mantiene almeno 1D
            #audio_tensor = waveform.squeeze(0) if waveform.dim() > 1 else waveform
            audio_tensor = waveform
            print("DEBUG: Waveform after squeeze/fix shape:", audio_tensor.shape)

            # --- Estrai embedding ---
            text_emb = model.get_query_embed(modality='text', text=[prompt_text], device=device)
            audio_emb = model.get_query_embed(modality='audio', audio=audio_tensor.to(device), device=device)

            # --- Calcola CLAP score (dot product o cosine similarity) ---
            score = (text_emb * audio_emb).sum(-1).item()

            scores_dict[filename] = score
            scores_list.append(score)

    avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
    print(f"Average CLAPScore: {avg_score:.4f}")

    return avg_score

def get_dataset_distribution(folder_path, model):
    all_probs = []
    valid_extensions = ('.wav', '.mp3', '.MP3', '.WAV')
    files = [f for f in os.listdir(folder_path) if f.endswith(valid_extensions)]

    if not files:
        raise ValueError(f"Nessun file audio (.wav, .mp3) trovato in {folder_path}")

    model.to(Config.DEVICE)
    model.eval()

    for file_name in files:
        path = os.path.join(folder_path, file_name)

        try:
            waveform, sr = torchaudio.load(path)

            target = 32000
            if sr != target:
                resampler = torchaudio.transforms.Resample(sr, target)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if waveform.shape[1] < target:
                waveform = F.pad(waveform, (0, target - waveform.shape[1]))
            else:
                waveform = waveform[:, :target]

            with torch.no_grad():
                logits = model(waveform.to(Config.DEVICE))
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        except Exception as e:
            print(f"Errore nel processare {file_name}: {e}")
            continue

    if not all_probs:
        return None

    return np.mean(np.vstack(all_probs), axis=0)









def plot_fad(real_embeddings, gen_embeddings):
    pca = PCA(n_components=2)

    combined = np.vstack((real_embeddings, gen_embeddings))
    reduced = pca.fit_transform(combined)

    n_real = len(real_embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:n_real, 0], reduced[:n_real, 1], label='Real/Remain Set', alpha=0.5)
    plt.scatter(reduced[n_real:, 0], reduced[n_real:, 1], label='Generated/Unlearned', alpha=0.5)
    plt.title("Visualizzazione degli Embeddings (FAD Analysis)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"audio_out/tx2m/20260210-1002_FT_100artists_FAD.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_kld(p_dist, q_dist):

    plt.figure(figsize=(10, 6))

    bins = np.arange(len(p_dist))

    plt.fill_between(bins, p_dist, alpha=0.4, label='Real Distribution (P)', color='royalblue')
    plt.plot(bins, p_dist, color='blue', lw=2)
    plt.fill_between(bins, q_dist, alpha=0.4, label='Generated Distribution (Q)', color='orange')
    plt.plot(bins, q_dist, color='darkorange', lw=2)

    plt.title('Comparison of Distributions (P vs Q)', fontsize=14)
    plt.xlabel('Logit / Bin Index', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    sns.despine()
    plt.savefig(f"audio_out/tx2m/20260210-1002_FT_100artists_KLD.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_clap(scores, filenames=None):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    if filenames and len(scores) < 20:
        # GRAFICO A BARRE (se i file sono pochi, es. meno di 20)
        sns.barplot(x=scores, y=filenames, hue=filenames, palette="viridis", legend=False)
        plt.xlabel("CLAP Score (Similarity)")
        plt.title("CLAP Score per ogni file audio")
        plt.xlim(0, 1) # La similarità va da -1 a 1, ma i valori buoni sono sopra 0
    else:
        # ISTOGRAMMA (se hai molti file, per vedere la distribuzione)
        sns.histplot(scores, kde=True, color="skyblue", bins=15)
        plt.axvline(np.mean(scores), color='red', linestyle='--', label=f'Media: {np.mean(scores):.3f}')
        plt.xlabel("CLAP Score")
        plt.ylabel("Numero di file")
        plt.title("Distribuzione dei CLAP Score nella cartella")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"audio_out/tx2m/20260210-1002_FT_100artists_CLAP.png", dpi=300, bbox_inches='tight')
    plt.show()