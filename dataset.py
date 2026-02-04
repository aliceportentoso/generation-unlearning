import torch
import torchaudio
import os
from torch.utils.data import Dataset
from config import *


class FMADataset(Dataset):
    def __init__(self, track_ids, metadata_df, target_samples=None):
        """
        track_ids: ID dei brani
        labels: Etichette numeriche (opzionali per Stable Audio, ma utili per coerenza)
        metadata_df: Il DataFrame 'tracks' di FMA che contiene le descrizioni
        target_samples: Numero di campioni (Config.SAMPLE_RATE * Config.DURATION)
        """
        self.track_ids = track_ids
        self.metadata = metadata_df
        self.target_samples = target_samples or (Config.SAMPLE_RATE * Config.DURATION)

    def __len__(self):
        return len(self.track_ids)

    def get_prompt_from_fma(self, track_id):
        """
        Recupera informazioni dal dataset FMA per creare un prompt.
        Esempio: 'Rock song by ArtistName from AlbumName'
        """
        try:
            row = self.metadata.loc[track_id]
            genre = row[('track', 'genre_top')]
            title = row[('track', 'title')]
            artist = row[('artist', 'name')]
            return f"{genre} song, {title} by {artist}"
        except:
            return "Audio track"

    def _load_waveform(self, filepath):
        waveform, sr = torchaudio.load(filepath)

        # 1. Resampling al sample rate del modello (es. 44100)
        if sr != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # 2. Mixdown a Stereo o Mono in base a cosa vuole Stable Audio
        # Stable Audio Open di solito è Stereo (2 canali)
        if waveform.shape[0] > 2:
            waveform = waveform[:2, :]
        elif waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # 3. Taglio o Padding
        if waveform.shape[1] > self.target_samples:
            waveform = waveform[:, :self.target_samples]
        elif waveform.shape[1] < self.target_samples:
            padding = self.target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]

        # Costruzione path FMA
        tid_str = f"{track_id:06d}"
        folder = tid_str[:3]
        filepath = os.path.join(Config.AUDIO_DIR, folder, tid_str + '.mp3')

        # Caricamento Waveform (NON Mel-Spectrogram)
        try:
            waveform = self._load_waveform(filepath)
        except Exception as e:
            print(f"Errore caricamento {filepath}: {e}")
            # Ritorna un tensore di zero se il file è corrotto per non fermare il batch
            waveform = torch.zeros((2, self.target_samples))

        # Recupero del prompt per il conditioning
        prompt = self.get_prompt_from_fma(track_id)

        # Ritorna: Waveform per la loss, il prompt per il condizionamento, e l'ID
        return waveform, prompt