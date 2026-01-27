import os
import json
import torch
import torchaudio
from pathlib import Path
from stable_audio_tools.models.pretrained import get_pretrained_model
import numpy as np

def load_audio(path, sample_rate=44100):
    audio, sr = torchaudio.load(path)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.unsqueeze(0)  # (1, C, T)

def main(input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stable Audio Open
    model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    dac = model.pretransform
    dac.eval().requires_grad_(False)

    for file in Path(input_dir).rglob("*.wav"):
        audio = load_audio(str(file))
        with torch.no_grad():
            latents = dac.encode(audio)

        out = output_dir / (file.stem + ".npy")
        np.save(out, latents.cpu().numpy())

if __name__ == "__main__":
    main("../data/fma_minimum", "../data/fma_latents")
