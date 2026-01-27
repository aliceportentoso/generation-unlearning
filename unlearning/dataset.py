from collections import Counter
from torch.utils.data import Dataset
import torchaudio
import os
from config import *

# Crea la trasformazione MelSpectrogram globale
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=Config.SAMPLE_RATE,
    n_fft=Config.WINDOW_SIZE,
    hop_length=Config.HOP_SIZE,
    n_mels=Config.N_MELS
)

def preprocessing(filepath, num_samples=Config.SAMPLE_RATE * Config.DURATION):
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception as e:
        print("➡️ Dettagli errore:", e)
        raise e  # rilancia l’eccezione per fermare il training

    # - converti in mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform[0]

    # - taglia o fai padding
    if waveform.shape[0] > num_samples:
        waveform = waveform[:num_samples]
    elif waveform.shape[0] < num_samples:
        padding = num_samples - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # - converti in Mel Spectrogram
    mel_spec = mel_spec_transform(waveform)  # [n_mels, time]
    mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

    return mel_spec

class FMADataset(Dataset):
    def __init__(self, track_ids, labels, augmenter = None, min_count = 100):
        self.track_ids = track_ids
        self.labels = labels
        self.num_samples = Config.SAMPLE_RATE * Config.DURATION
        self.augmenter = augmenter

        # data augmentation per le classi molto poco rappresentate
        if self.augmenter is not None:
            counter = Counter(labels)
            self.track_ids = []
            self.labels = []

            for idx, label in enumerate(labels):
                self.track_ids.append(track_ids[idx])
                self.labels.append(label)

                if counter[label] < min_count:
                    n_repeat = (min_count - counter[label])/2
                    for x in range(int(n_repeat)):
                        self.track_ids.append(track_ids[idx])
                        self.labels.append(label)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        label = self.labels[idx]

        tid_str = f"{track_id:06d}"
        folder = tid_str[:3]
        filepath = os.path.join(Config.AUDIO_DIR, folder, tid_str + '.mp3')

        mel_spec = preprocessing(filepath, self.num_samples)

        if self.augmenter is not None and idx >= len(self.track_ids):
            mel_spec = self.augmenter(mel_spec)

        return mel_spec, int(label)
