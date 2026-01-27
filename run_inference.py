import argparse
from stable_audio_tools.inference.inference import generate_audio, save_audio, free_memory
from stable_audio_tools.models.local_pretrained import get_pretrained_model_local

CONFIG_JSON = "./stable_audio_tools/configs/model_configs/txt2audio/stable_audio_1_0.json"

args = argparse.Namespace(
    prompt="",
    cfg=7.0,
    steps=200,
    model_ckpt="stable-audio-open-1.0.ckpt",
    seed=-1,
    start_sec=0.0,
    duration=47.0,
    output="output.wav"
)

model, model_config = get_pretrained_model_local(
    CONFIG_JSON,
    args.model_ckpt
)

print("Model loaded")

audio, sr = generate_audio(
    model,
    prompt=args.prompt,
    cfg_scale=args.cfg,
    steps=args.steps,
    start_sec=args.start_sec,
    duration_sec=args.duration,
    seed=args.seed
)

print("End of generation. Saving audio...")

save_audio(args.output, audio, sr)
free_memory()