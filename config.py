from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Config:

    SUBSET = "medium"
    DEVICE = "cuda"

    EPOCHS = 1
    LR = 1e-5 #1e-3
    NUM_ARTISTS = 1

    # Unlearning
    UNL_METHOD = "FT"   # FT, GA, ST, OSM, A

    AUDIO_DIR = "../data/fma_large"
    CSV_FILE = "../data/tracks.csv"

    # Audio parameters
    SAMPLE_RATE = 22050
    WINDOW_SIZE = 1024
    HOP_SIZE = 512
    N_MELS = 64
    NUM_CLASSES = 8
    NUM_FRAMES = 1292
    DURATION = 30
    BATCH_SIZE = 8
    fmin = 0
    fmax = SAMPLE_RATE // 2

    @classmethod
    def print_config_unl(cls):
        print("---- UNLEARNING CONFIG ----")
        print(f"Epochs         : {cls.EPOCHS}")
        print(f"Learning rate  : {cls.LR}")
        print(f"Dataset SUBSET : {cls.SUBSET}")
        print(f"METHOD         : {cls.UNL_METHOD}")

