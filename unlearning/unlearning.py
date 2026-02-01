from diffusers import AutoencoderOobleck
from torch import optim

import torchaudio
import torch

from config import Config
from stable_audio_tools.models.autoencoders import AudioAutoencoder

from stable_audio_tools.models.autoencoders import create_autoencoder_from_config


def unl_fine_tuning(model, forget_loader, retain_loader, model_config, epochs=1, lr=5e-6):
    """
    Esegue l'unlearning bilanciato.
    - Massimizza la loss sui dati da dimenticare (forget_set)
    - Minimizza la loss sui dati da mantenere (retain_set)
    """
    device = next(model.parameters()).device
    # Stable Audio Open richiede spesso l'uso del modulo precision corretto
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    sample_rate = model_config.get("sample_rate", 44100)
    # Il sample_size serve per definire la lunghezza dei tensori audio
    sample_size = model_config.get("sample_size", 65536)

    model.train()

    # 1️⃣ Carica il checkpoint con i pesi del VAE
    ckpt_path = "vae_model.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 2️⃣ Config del tuo VAE (dal JSON)
    vae_config = {
        "audio_channels": 2,
        "channel_multiples": [1, 2, 4, 8, 16],
        "decoder_channels": 128,
        "decoder_input_channels": 64,
        "downsampling_ratios": [2, 4, 4, 8, 8],
        "encoder_hidden_size": 128,
        "sampling_rate": 44100
    }

    # 3️⃣ Crea direttamente l'autoencoder (senza create_autoencoder_from_config)
    autoencoder = AutoencoderOobleck(
        audio_channels=vae_config["audio_channels"],
        channel_multiples=vae_config["channel_multiples"],
        decoder_channels=vae_config["decoder_channels"],
        decoder_input_channels=vae_config["decoder_input_channels"],
        downsampling_ratios=vae_config["downsampling_ratios"],
        encoder_hidden_size=vae_config["encoder_hidden_size"],
        sampling_rate=vae_config["sampling_rate"]
    )

    # 4️⃣ Carica i pesi
    autoencoder.load_state_dict(ckpt['state_dict'], strict=False)

    # 5️⃣ Modalità eval e invio a device
    autoencoder.eval().to(device)

    for epoch in range(epochs):
        model.train()

        for batch_idx, (waveforms, prompts, _) in enumerate(forget_loader):
            optimizer.zero_grad()

            waveforms = waveforms.to(device)  # (B, 2, T)
            batch_size = waveforms.shape[0]

            #conditioner
            if isinstance(prompts, str):
                prompts = [prompts]

            input_data = {
                "prompt": list(prompts),
                "seconds_start": [0] * batch_size,
                "seconds_total": [30] * batch_size
            }

            try:
                with torch.no_grad():
                    cond = model.conditioner(input_data, device=device)
            except TypeError:
                batch_list = []
                for i in range(batch_size):
                    batch_list.append({
                        "prompt": prompts[i],
                        "seconds_start": 0,
                        "seconds_total": 30
                    })
                with torch.no_grad():
                    cond = model.conditioner(batch_list, device=device)

            #encoding
            with torch.no_grad():
                posterior = autoencoder.encode(waveforms)

                latent_dist = posterior.latent_dist
                latents_mean = latent_dist.mean
                latents = latents_mean * 0.18215 #latent scale

            #timestamp
            t = torch.rand(batch_size, device=device)

            loss = model(latents, t=t, cond=cond).mean()

            (-loss).backward()
            optimizer.step()


            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Forget Loss: {loss.item():.4f}")

    print("Fine-tuning di Unlearning completato.")
    return model


def load_audio_tensor(track_id, target_samples):
    # 1. Costruisci il path (adatta questo al tuo filesystem FMA)
    # Esempio: "data/fma_small/000/000140.mp3"
    path = Config.get_audio_path(track_id)

    # 2. Carica l'audio
    waveform, sr = torchaudio.load(path)

    # 3. Resampling se necessario (Stable Audio Open lavora a 44.1k o 48k)
    if sr != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # 4. Padding o Truncate per arrivare a target_samples
    if waveform.size(1) > target_samples:
        waveform = waveform[:, :target_samples]  # Taglia
    else:
        padding = target_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding))  # Allunga con silenzio

    # Restituisce con dimensione batch: [1, canali, campioni]
    return waveform.unsqueeze(0).to(Config.DEVICE)