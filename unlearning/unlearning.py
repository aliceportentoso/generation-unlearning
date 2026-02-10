import torch
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderOobleck
import matplotlib.pyplot as plt
from config import Config

def setup_lora(model, lr):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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

def get_conditioning_and_latents(model, autoencoder, waveforms, prompts):
    device = Config.DEVICE
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
        batch_list = [{"prompt": p, "seconds_start": 0, "seconds_total": 30} for p in prompts_list]
        cond = model.conditioner(batch_list, device=device)

    with torch.no_grad(): # per risparmiare memoria
        posterior = autoencoder.encode(waveforms)
        latents = posterior.latent_dist.mean * 0.18215

    return cond, latents

def unl_fine_tuning(model, forget_loader, retain_loader, epochs, lr, lambda_unlearn):
    device = next(model.parameters()).device
    model, optimizer = setup_lora(model, lr)
    autoencoder = load_vae(device)
    model.train()

    history_f, history_r = [], []

    for epoch in range(epochs):
        retain_iter = iter(retain_loader)
        total_f, total_r, count = 0, 0, 0

        for batch_forget in forget_loader:
            try:
                batch_retain = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch_retain = next(retain_iter)

            w_f, p_f = batch_forget
            w_r, p_r = batch_retain
            w_f, w_r = w_f.to(device), w_r.to(device)

            # Estrazione latenti e condizionamento
            cond_r, lat_r = get_conditioning_and_latents(model, autoencoder, w_r, p_r)
            cond_f, lat_f = get_conditioning_and_latents(model, autoencoder, w_f, p_f)

            t_f = torch.rand(lat_f.shape[0], device=device)
            t_r = torch.rand(lat_r.shape[0], device=device)

            # Calcolo delle Loss
            loss_f = torch.nn.functional.mse_loss(model(lat_f, t=t_f, cond=cond_f), lat_f)
            loss_r = torch.nn.functional.mse_loss(model(lat_r, t=t_r, cond=cond_r), lat_r)
            loss = loss_r - lambda_unlearn * loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_f += loss_f.item()
            total_r += loss_r.item()
            count += 1

        history_f.append(total_f / count)
        history_r.append(total_r / count)
        print(f"Epoca {epoch + 1} | Retain Loss: {total_r / count:.4f} | Forget Loss: {total_f / count:.4f}")

    # --- Generazione del grafico finale ---
    plt.figure(figsize=(8, 4))
    plt.plot(history_r, label='Retain Loss (Qualità)', color='blue', marker='o')
    plt.plot(history_f, label='Forget Loss (Dimenticati)', color='red', marker='x')
    plt.title("Andamento Unlearning")
    plt.xlabel("Epoca")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"audio_out/tx2m/20260210-1002_FT_100artists_Loss.png", dpi=300, bbox_inches='tight')
    plt.show()

    return model

def unl_gradient_ascent(model, forget_loader, retain_loader, epochs, lr, alpha=1, beta=1):
    device = next(model.parameters()).device
    model, optimizer = setup_lora(model, lr)
    autoencoder = load_vae(device)
    model.train()

    for epoch in range(epochs):
        retain_iter = iter(retain_loader)
        total_f, total_r, count = 0, 0, 0

        for batch_forget in forget_loader:
            try:
                batch_retain = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch_retain = next(retain_iter)

            w_f, p_f = batch_forget
            w_r, p_r = batch_retain
            w_f, w_r = w_f.to(device), w_r.to(device)

            cond_f, lat_f = get_conditioning_and_latents(model, autoencoder, w_f, p_f)
            cond_r, lat_r = get_conditioning_and_latents(model, autoencoder, w_r, p_r)

            t_f = torch.rand(lat_f.shape[0], device=device)
            t_r = torch.rand(lat_r.shape[0], device=device)

            # Calcolo losses
            loss_forget = torch.nn.functional.mse_loss(model(lat_f, t=t_f, cond=cond_f), lat_f)
            loss_retain = torch.nn.functional.mse_loss(model(lat_r, t=t_r, cond=cond_r), lat_r)
            loss = alpha * loss_retain - beta * loss_forget

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_f += loss_forget.item()
            total_r += loss_retain.item()
            count += 1

        print(f"Epoch {epoch + 1}/{epochs} | Retain Loss (Minimizing): {total_r / count:.4f} | Forget Loss (Maximizing): {total_f / count:.4f}")

    return model

def unl_stochastic_teacher(model, forget_loader, retain_loader, epochs, lr, alpha=1, beta=0.5):
    device = next(model.parameters()).device

    # 1. Configura LoRA sul modello originale
    model, optimizer = setup_lora(model, lr)
    autoencoder = load_vae(device)

    for epoch in range(epochs):
        model.train()

        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        num_batches = min(len(retain_loader), len(forget_loader))

        for batch_idx in range(num_batches):
            try:
                f_waveforms, f_prompts = next(forget_iter)
                r_waveforms, r_prompts = next(retain_iter)
            except StopIteration:
                break

            optimizer.zero_grad()
            # --- FORGET STEP (Logica Teacher-Student Inversa) ---
            f_waveforms = f_waveforms.to(device)
            with torch.no_grad():
                f_cond, f_latents = get_conditioning_and_latents(model, autoencoder, f_waveforms, f_prompts)
                f_t = torch.rand(f_waveforms.shape[0], device=device)

                # Otteniamo la predizione "giusta" dal modello originale (Teacher)
                with model.disable_adapter():
                    teacher_preds = model(f_latents, t=f_t, cond=f_cond)

            # Predizione attuale (Student con LoRA)
            student_preds = model(f_latents, t=f_t, cond=f_cond)

            forget_loss = -torch.nn.functional.mse_loss(student_preds, teacher_preds)

            # --- RETAIN STEP (Logica Standard Fine-tuning) ---
            r_waveforms = r_waveforms.to(device)
            r_cond, r_latents = get_conditioning_and_latents(model, autoencoder, r_waveforms, r_prompts)
            r_t = torch.rand(r_waveforms.shape[0], device=device)

            # Predizione sui dati da mantenere
            retain_preds = model(r_latents, t=r_t, cond=r_cond)

            retain_loss = torch.nn.functional.mse_loss(retain_preds, r_latents)

            # --- COMBINAZIONE DELLE LOSS ---
            loss = alpha * retain_loss + beta * forget_loss

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Retain Loss: {retain_loss.item():.4f} | Forget Loss: {forget_loss.item():.4f}")

    return model

def unl_one_shot_magnitude(model, threshold=0.1):
    model, _ = setup_lora(model, lr=1e-5)  # Inizializziamo LoRA

    model.eval()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                # Creiamo una maschera per i pesi con magnitudo elevata
                mask = torch.abs(param) < (param.max() * threshold)
                param.mul_(mask)  # Azzera i pesi sopra la soglia

    print(f"One-shot Magnitude Pruning completato (Soglia: {threshold}).")
    return model


def unl_amnesiac(model, forget_loader, lr):
    device = next(model.parameters()).device
    model, optimizer = setup_lora(model, lr)
    autoencoder = load_vae(device)
    model.train()

    for batch_idx, (waveforms, prompts) in enumerate(forget_loader):
        optimizer.zero_grad()
        waveforms = waveforms.to(device)

        with torch.no_grad():
            cond, latents = get_conditioning_and_latents(model, autoencoder, waveforms, prompts)
            target_noise = torch.randn_like(latents)

        output = model(latents, t=torch.rand(waveforms.shape[0], device=device), cond=cond)
        loss = torch.nn.functional.mse_loss(output, target_noise)

        loss.backward()
        optimizer.step()

        if batch_idx >= 5: break

    print("Amnesiac Unlearning completato.")
    return model
