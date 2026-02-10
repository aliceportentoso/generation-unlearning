import torch.nn.functional as F
from hear21passt.base import get_basic_model
from scipy.linalg import sqrtm
import torchaudio
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

import os
import torch
import numpy as np
import laion_clap
from torch.nn.functional import cosine_similarity

def get_embeddings(model, folder_path):
    all_embeddings = []
    TARGET_SR = 16000
    model.to(Config.DEVICE)
    model.eval()

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3'))]
    for file in tqdm(files): #, desc=f"Extracting from {os.path.basename(folder_path)}"):
        path = os.path.join(folder_path, file)
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:  # Mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)

            audio_input = waveform.squeeze().numpy()

            with torch.no_grad():
                emb_frames = model.forward(audio_input, fs=TARGET_SR)

                if torch.is_tensor(emb_frames):
                    emb_frames = emb_frames.cpu().numpy()

                if emb_frames.ndim > 1:
                    emb_avg = np.mean(emb_frames, axis=0)
                else:
                    emb_avg = emb_frames

                all_embeddings.append(emb_avg)

        except Exception as e:
            print(f"Errore su {file}: {e}")

    return np.array(all_embeddings)

def compute_fad(real_path, gen_path):
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.to(Config.DEVICE)
    model.eval()

    real_embeddings = get_embeddings(model, real_path)
    gen_embeddings = get_embeddings(model, gen_path)
    plot_fad(real_embeddings, gen_embeddings)

    if len(real_embeddings) < 2 or len(gen_embeddings) < 2:
        return float('nan')

    mu_r = np.mean(real_embeddings, axis=0)
    sigma_r = np.cov(real_embeddings, rowvar=False)

    mu_g = np.mean(gen_embeddings, axis=0)
    sigma_g = np.cov(gen_embeddings, rowvar=False)

    diff = mu_r - mu_g
    mean_dist = diff.dot(diff)

    eps = 1e-6
    sigma_r += np.eye(sigma_r.shape[0]) * eps
    sigma_g += np.eye(sigma_g.shape[0]) * eps

    covmean, _ = sqrtm(sigma_r.dot(sigma_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad_score = mean_dist + np.trace(sigma_r + sigma_g - 2 * covmean)

    return fad_score

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

def compute_kld(real_path, gen_path):
    model = get_basic_model(mode="logits").to(Config.DEVICE)
    model.eval()

    p_dist = get_dataset_distribution(real_path, model)
    q_dist = get_dataset_distribution(gen_path, model)

    eps = 1e-7
    p_dist = np.clip(p_dist, eps, 1.0)
    q_dist = np.clip(q_dist, eps, 1.0)

    plot_kld(p_dist, q_dist)
    kld_score = np.sum(p_dist * (np.log(p_dist) - np.log(q_dist)))

    return kld_score

def compute_clap(folder_path):

    ckpt_path = "music_audioset_epoch_15_esc_90.14.pt"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=Config.DEVICE)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato in: {ckpt_path}")

    model.load_ckpt(ckpt_path)
    model.eval()

    scores = []

    with torch.no_grad():
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                prompt = filename.replace(".wav", "").replace("_", " ").replace("-", " ")
                audio_path = os.path.join(folder_path, filename)

                try:
                    text_embed = model.get_text_embedding([prompt])
                    audio_embed = model.get_audio_embedding_from_filelist(x=[audio_path])

                    t_tensor = torch.from_numpy(text_embed).to(Config.DEVICE)
                    a_tensor = torch.from_numpy(audio_embed).to(Config.DEVICE)

                    similarity = cosine_similarity(t_tensor, a_tensor).item()

                    scores.append(similarity)

                except Exception as e:
                    print(f"Errore durante l'analisi di {filename}: {e}")

    if not scores:
        print("Nessun file audio processato correttamente.")
        return 0.0

    avg_score = np.mean(scores)
    plot_clap(scores)

    return avg_score

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