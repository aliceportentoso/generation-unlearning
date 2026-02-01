import torch
import torchaudio
from torch import optim
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderOobleck
from config import Config

def setup_lora(model, lr):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    return model, optimizer


def load_vae(device):
    ckpt_path = "vae_model.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vae_config = {
        "audio_channels": 2,
        "channel_multiples": [1, 2, 4, 8, 16],
        "decoder_channels": 128,
        "decoder_input_channels": 64,
        "downsampling_ratios": [2, 4, 4, 8, 8],
        "encoder_hidden_size": 128,
        "sampling_rate": 44100
    }
    autoencoder = AutoencoderOobleck(**vae_config)
    autoencoder.load_state_dict(ckpt['state_dict'], strict=False)
    return autoencoder.eval().to(device)


def get_conditioning_and_latents(model, autoencoder, waveforms, prompts, device):
    batch_size = waveforms.shape[0]
    prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)

    input_data = {
        "prompt": prompts_list,
        "seconds_start": [0] * batch_size,
        "seconds_total": [30] * batch_size
    }

    try:
        cond = model.conditioner(input_data, device=device)
    except TypeError:
        # Fallback per formati batch alternativi
        batch_list = [{"prompt": p, "seconds_start": 0, "seconds_total": 30} for p in prompts_list]
        cond = model.conditioner(batch_list, device=device)

    posterior = autoencoder.encode(waveforms)
    latents = posterior.latent_dist.mean * 0.18215
    return cond, latents

def unl_fine_tuning(model, forget_loader, epochs, lr):
    device = next(model.parameters()).device
    model, optimizer = setup_lora(model, lr)
    autoencoder = load_vae(device)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (waveforms, prompts, _) in enumerate(forget_loader):
            optimizer.zero_grad()
            waveforms = waveforms.to(device)

            with torch.no_grad():
                cond, latents = get_conditioning_and_latents(model, autoencoder, waveforms, prompts, device)

            t = torch.rand(waveforms.shape[0], device=device)
            loss = model(latents, t=t, cond=cond).mean()

            (-loss).backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Ascent (Forget) Loss: {loss.item():.4f}")

    return model


def unl_gradient_ascent(model, forget_loader, retain_loader, model_config, epochs, lr, alpha=1.5, beta=1.0):
    device = next(model.parameters()).device
    model, optimizer = setup_lora(model, lr)
    vae = load_vae(device)
    retain_iter = iter(retain_loader)

    for epoch in range(epochs):
        model.train()
        for waveforms_f, prompts_f, _ in forget_loader:
            optimizer.zero_grad()

            # --- FORGET (Gradient Ascent) ---
            cond_f, latents_f = get_conditioning_and_latents(model, vae, waveforms_f.to(device), prompts_f, device)
            loss_forget = model(latents_f, t=torch.rand(latents_f.shape[0], device=device), cond=cond_f).mean()

            # --- RETAIN (Gradient Descent) ---
            try:
                waveforms_r, prompts_r, _ = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                waveforms_r, prompts_r, _ = next(retain_iter)

            cond_r, latents_r = get_conditioning_and_latents(model, vae, waveforms_r.to(device), prompts_r, device)
            loss_retain = model(latents_r, t=torch.rand(latents_r.shape[0], device=device), cond=cond_r).mean()

            # Bilanciamento: Minimizza retain, Massimizza forget (segno meno)
            loss = beta * loss_retain - (alpha * loss_forget)
            loss.backward()
            optimizer.step()

    # Merge finale per risolvere il problema del 'cond' in inferenza
    return model.merge_and_unload()

def unl_stochastic_teacher(model, forget_loader, epochs, lr):
    device = next(model.parameters()).device

    # 1. Configura LoRA sul modello originale
    model, optimizer = setup_lora(model, lr)
    autoencoder = load_vae(device)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (waveforms, prompts, _) in enumerate(forget_loader):
            optimizer.zero_grad()
            waveforms = waveforms.to(device)

            with torch.no_grad():
                cond, latents = get_conditioning_and_latents(model, autoencoder, waveforms, prompts, device)
                t = torch.rand(waveforms.shape[0], device=device)

                # TEACHER MODE : disattiviamo temporaneamente LoRA per ottenere la predizione "originale"
                with model.disable_adapter():
                    teacher_preds = model(latents, t=t, cond=cond)

            #  STUDENT MODE ---
            # Calcoliamo la predizione con LoRA attivo
            student_preds = model(latents, t=t, cond=cond)

            loss = torch.nn.functional.mse_loss(student_preds, teacher_preds)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Stochastic Teacher Loss: {loss.item():.4f}")
    return model

def unl_one_shot_magnitude(model, threshold=0.1):
    """
    Identifica i pesi LoRA che si attivano maggiormente sui dati forget
    e li azzera (pruning) per rimuovere l'informazione.
    """
    device = next(model.parameters()).device
    model, _ = setup_lora(model, lr=1e-5)  # Inizializziamo LoRA

    model.eval()
    # In questa tecnica, identifichiamo i parametri LoRA e azzeriamo una % dei pesi
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                # Creiamo una maschera per i pesi con magnitudo elevata
                mask = torch.abs(param) < (param.max() * threshold)
                param.mul_(mask)  # Azzera i pesi sopra la soglia

    print(f"One-shot Magnitude Pruning completato (Soglia: {threshold}).")
    return model


def unl_amnesiac(model, forget_loader, lr):
    """
    Esegue un 'relabeling' dei dati forget con target casuali o
    semplicemente inverte la direzione del gradiente in un singolo batch massiccio.
    """
    device = next(model.parameters()).device
    model, optimizer = setup_lora(model, lr)
    autoencoder = load_vae(device)
    model.train()

    for batch_idx, (waveforms, prompts, _) in enumerate(forget_loader):
        optimizer.zero_grad()
        waveforms = waveforms.to(device)

        # Generiamo target casuali (Amnesiac) invece di usare quelli del modello
        with torch.no_grad():
            cond, latents = get_conditioning_and_latents(model, autoencoder, waveforms, prompts, device)
            # Creiamo un rumore target che il modello "dovrebbe" aver predetto
            # se non avesse mai visto questi dati
            target_noise = torch.randn_like(latents)

            # Il modello cerca di predire il rumore casuale invece delle feature reali
        output = model(latents, t=torch.rand(waveforms.shape[0], device=device), cond=cond)
        loss = torch.nn.functional.mse_loss(output, target_noise)

        loss.backward()
        optimizer.step()

        # Di solito Amnesiac richiede pochissimi step (spesso uno solo per batch)
        if batch_idx >= 5: break

    print("Amnesiac Unlearning completato.")
    return model

def load_audio_tensor(track_id, target_samples):
    path = Config.get_audio_path(track_id)
    waveform, sr = torchaudio.load(path)

    if sr != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.size(1) > target_samples:
        waveform = waveform[:, :target_samples]
    else:
        padding = target_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform.unsqueeze(0).to(Config.DEVICE)