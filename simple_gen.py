import json
import torch
import torchaudio
from einops import rearrange

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict

from datetime import datetime

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

conditioning = [{
    "prompt": "rock song with intro, verse, chorus, guitar solo",
    "seconds_start": 0,
    "seconds_total": 30  # durata in secondi
}]

my_model = False
if my_model:
    #name = "stable_audio_tools/yzxnx12q/checkpoints/UNWRAPPED_epoch=0-step=10000.ckpt.safetensors"
    name = f"audio_out/fine_tuning/{timestamp}_{conditioning[0]["prompt"].replace(" ", "_")}"

    with open("stable_audio_tools/configs/model.json") as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    copy_state_dict(
        model,
        load_ckpt_state_dict(
            name
        )
    )

else:
    from stable_audio_tools import get_pretrained_model

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    name = f"audio_out/pretrain_model/{timestamp}_{conditioning[0]["prompt"].replace(" ", "_")}"

device = "cuda" if torch.cuda.is_available() else "cpu"
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