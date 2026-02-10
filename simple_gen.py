import json
import torch
import torchaudio
from einops import rearrange

from config import Config
from stable_audio_tools.inference.generation import generate_diffusion_cond

from datetime import datetime
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

def generate(prompt):
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": 30  # durata in secondi
    }]

    from stable_audio_tools import get_pretrained_model

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    name = f"audio_out/pretrain_model/{timestamp}_{conditioning[0]['prompt'].replace('', '_')}"

    device = Config.DEVICE
    model = model.to(device)

    with torch.no_grad():
        audio = generate_diffusion_cond(
            model=model,
            conditioning=conditioning,
            steps=100,           # numero di passi della diffusion
            cfg_scale=7.0,       # scala CFG (più alto = più fedele al prompt)
            sampler_type="dpmpp-3m-sde",  # tipo di sampler
            device=device,
            sample_size=model_config["sample_size"]
        )


    audio = rearrange(audio, "b c t -> c (b t)")

    # norm
    audio = audio / audio.abs().max()

    torchaudio.save(
        f"{name}.wav",
        audio.cpu(),
        sample_rate=model_config["sample_rate"]
    )

    print(f"Audio generato in {name}.wav")

generate("crete a song in the style of Nicky Cook")