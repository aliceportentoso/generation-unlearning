import laion_clap
import librosa
import torch.nn.functional as F
from hear21passt.base import get_basic_model
from scipy.linalg import sqrtm
import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm


def get_embeddings(model, folder_path, device="cuda"):
    all_embeddings = []
    TARGET_SR = 16000
    model.to(device)
    model.eval()

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3'))]

    for file in tqdm(files, desc=f"Extracting from {os.path.basename(folder_path)}"):
        path = os.path.join(folder_path, file)
        try:
            # 1. Caricamento e Resampling
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:  # Mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)

            # 2. Preparazione per VGGish
            audio_input = waveform.squeeze().numpy()

            with torch.no_grad():
                # L'output di VGGish è tipicamente (N_frames, 128)
                emb_frames = model(audio_input)

                # Se il modello restituisce un tensore PyTorch, lo portiamo su CPU/NumPy
                if torch.is_tensor(emb_frames):
                    emb_frames = emb_frames.cpu().numpy()

                # 3. Aggregazione per FAD: Media di tutti i frame del brano
                # Questo trasforma (N_frames, 128) in un singolo vettore (128,)
                if emb_frames.ndim > 1:
                    emb_avg = np.mean(emb_frames, axis=0)
                else:
                    emb_avg = emb_frames

                all_embeddings.append(emb_avg)

        except Exception as e:
            print(f"Errore su {file}: {e}")

    return np.array(all_embeddings)

def compute_fad(real_path, gen_path, device="cuda"):
    """
    Calcola la Fréchet Audio Distance tra due cartelle di file audio.
    """
    # 1. Caricamento Modello VGGish
    print("Inizializzazione VGGish...")
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.to(device)
    model.eval()

    # 2. Estrazione Embedding (usando la funzione che abbiamo corretto)
    print("Analisi brani REALI...")
    real_embeddings = get_embeddings(model, real_path, device=device)

    print("Analisi brani GENERATI...")
    gen_embeddings = get_embeddings(model, gen_path, device=device)

    # Verifica minima per la statistica
    if len(real_embeddings) < 2 or len(gen_embeddings) < 2:
        return float('nan')  # Non si può calcolare la covarianza con un solo file

    # 3. Calcolo FAD
    mu_r = np.mean(real_embeddings, axis=0)
    sigma_r = np.cov(real_embeddings, rowvar=False)

    mu_g = np.mean(gen_embeddings, axis=0)
    sigma_g = np.cov(gen_embeddings, rowvar=False)

    # Distanza tra le medie
    diff = mu_r - mu_g
    mean_dist = diff.dot(diff)

    # Distanza tra le covarianze (Traccia)
    # Aggiungiamo un piccolo valore alla diagonale per stabilità numerica (eps)
    eps = 1e-6
    sigma_r += np.eye(sigma_r.shape[0]) * eps
    sigma_g += np.eye(sigma_g.shape[0]) * eps

    covmean, _ = sqrtm(sigma_r.dot(sigma_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad_score = mean_dist + np.trace(sigma_r + sigma_g - 2 * covmean)

    return fad_score

def compute_fad2(real_path, gen_path):
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.eval()

    mu_r, sigma_r = get_embeddings(model, real_path)
    mu_g, sigma_g = get_embeddings(model, gen_path)

    # Formula di Fréchet
    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r.dot(sigma_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)
    return fad

def get_dataset_distribution(folder_path, model, device="cuda"):
    all_probs = []
    # Lista solo i file .wav nella cartella
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not files:
        raise ValueError(f"Nessun file .wav trovato in {folder_path}")

    for file_name in files:
        path = os.path.join(folder_path, file_name)

        # Resampling, downmix a mono
        waveform, sr = torchaudio.load(path)
        if sr != 32000:
            resampler = torchaudio.transforms.Resample(sr, 32000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        target = 320000
        if waveform.shape[1] < target:
            waveform = F.pad(waveform, (0, target - waveform.shape[1]))
        else:
            waveform = waveform[:, :target]

        with torch.no_grad():
            logits = model(waveform.to(device))
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    # Calcolo della distribuzione media (vettore di 527 classi)
    return np.mean(np.vstack(all_probs), axis=0)

def compute_kld(real_path, gen_path, device="cuda"):
    print("Caricamento modello PaSST...")
    model = get_basic_model(mode="logits").to(device)
    model.eval()

    print(f"Analisi dataset reale: {real_path}")
    p_dist = get_dataset_distribution(real_path, model, device)

    print(f"Analisi dataset generato: {gen_path}")
    q_dist = get_dataset_distribution(gen_path, model, device)

    eps = 1e-7
    p_dist = np.clip(p_dist, eps, 1.0)
    q_dist = np.clip(q_dist, eps, 1.0)

    kld_score = np.sum(p_dist * (np.log(p_dist) - np.log(q_dist)))

    return kld_score


def compute_clap(gen_path, text_prompts, model_type='640t'):
    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt()

    # 1. Ottieni i file e ordinali per sicurezza
    files = sorted([os.path.join(gen_path, f) for f in os.listdir(gen_path) if f.endswith(('.wav', '.mp3'))])

    # Verifica che il numero di prompt coincida con il numero di file
    if len(files) != len(text_prompts):
        raise ValueError(f"Discrepanza: hai {len(files)} brani ma {len(text_prompts)} prompt!")

    similarities = []

    with torch.no_grad():
        # 2. Cicla contemporaneamente su file e prompt usando zip()
        for file_path, prompt in zip(files, text_prompts):
            # Carica l'audio
            audio_data, _ = librosa.load(file_path, sr=48000)

            # Calcola embedding Audio (x vuole una lista di array)
            audio_embed = model.get_audio_embedding_from_data(x=[audio_data])
            audio_embed = audio_embed / np.linalg.norm(audio_embed, axis=-1, keepdims=True)

            # Calcola embedding Testo (richiede lista di stringhe)
            # Usiamo [str(prompt)] per assicurarci che sia nel formato corretto
            text_embed = model.get_text_embedding([str(prompt)])
            text_embed = text_embed / np.linalg.norm(text_embed, axis=-1, keepdims=True)

            # Similarità tra la coppia brano-prompt
            sim = np.dot(text_embed, audio_embed.T)
            similarities.append(sim[0][0])

    # Restituisce la media delle similarità di tutte le coppie
    return np.mean(similarities)